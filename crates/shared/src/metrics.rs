//! Stage-level latency aggregation. Each pipeline stage emits a
//! `PipelineMetrics` after processing; this aggregator keeps a rolling
//! window of recent samples per stage and exposes percentiles for a UI
//! panel and tracing.
//!
//! Built deliberately small: a `VecDeque` per stage capped at
//! `DEFAULT_CAPACITY` samples, sorted on demand inside `percentile`.
//! Sort cost is `O(N log N)` where N is at most a few hundred — well
//! under a millisecond, called once per UI frame at most. Anything
//! fancier (online quantile sketches, t-digests) would be premature
//! optimisation for this scale.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::time::Duration;

/// Maximum number of recent samples retained per stage. Older samples
/// are evicted in FIFO order. 256 ≈ a couple of minutes of activity at
/// the current pipeline cadence — enough that the percentiles reflect
/// the *current* run, not the entire session, so a configuration
/// change shows up in the panel within ~1 minute.
pub const DEFAULT_CAPACITY: usize = 256;

/// Summary of a single stage's recent latency window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StageStats {
    pub p50: Duration,
    pub p95: Duration,
    /// Number of samples currently retained in the rolling window.
    /// Capped at the aggregator's capacity.
    pub window_size: usize,
    /// Total number of samples ever recorded for this stage. Survives
    /// window eviction so the UI can show "1234 samples seen".
    pub total_count: u64,
}

struct StageHistogram {
    samples: VecDeque<Duration>,
    total_count: u64,
}

impl StageHistogram {
    fn new(capacity: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(capacity),
            total_count: 0,
        }
    }

    fn record(&mut self, duration: Duration, capacity: usize) {
        if self.samples.len() == capacity {
            self.samples.pop_front();
        }
        self.samples.push_back(duration);
        self.total_count = self.total_count.saturating_add(1);
    }

    /// Compute the `p`-th percentile (0.0–100.0). Returns `None` when
    /// the window is empty.
    fn percentile(&self, p: f32) -> Option<Duration> {
        if self.samples.is_empty() {
            return None;
        }
        let mut sorted: Vec<Duration> = self.samples.iter().copied().collect();
        sorted.sort();

        let clamped = p.clamp(0.0, 100.0);
        let index = ((clamped / 100.0) * (sorted.len() as f32 - 1.0)).round() as usize;
        sorted.get(index).copied()
    }
}

/// Thread-safe aggregator. One per application; pipelines record into
/// it through any `Arc<StageMetricsAggregator>` they hold.
pub struct StageMetricsAggregator {
    inner: Mutex<HashMap<String, StageHistogram>>,
    capacity: usize,
}

impl StageMetricsAggregator {
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
            capacity: capacity.max(1),
        }
    }

    /// Record one observation for `stage`. Cheap: a `HashMap` lookup
    /// plus a `VecDeque` push (and possibly a pop_front when the
    /// window is full). Lock contention is the dominant cost.
    pub fn record(&self, stage: &str, duration: Duration) {
        let capacity = self.capacity;
        let mut inner = match self.inner.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        let histogram = inner
            .entry(stage.to_string())
            .or_insert_with(|| StageHistogram::new(capacity));
        histogram.record(duration, capacity);
    }

    /// Convenience: percentile for one stage. Returns `None` when the
    /// stage has no samples yet.
    pub fn percentile(&self, stage: &str, p: f32) -> Option<Duration> {
        let inner = self.inner.lock().ok()?;
        inner.get(stage).and_then(|h| h.percentile(p))
    }

    /// Stats for every stage. Stages with no samples are omitted (they
    /// would only contribute noise to the UI).
    pub fn snapshot(&self) -> HashMap<String, StageStats> {
        let inner = match self.inner.lock() {
            Ok(g) => g,
            Err(_) => return HashMap::new(),
        };

        let mut output = HashMap::with_capacity(inner.len());
        for (stage, histogram) in inner.iter() {
            let p50 = match histogram.percentile(50.0) {
                Some(v) => v,
                None => continue,
            };
            // Unwrap is safe here: percentile only returns None when the
            // window is empty, which the previous match already excluded.
            let p95 = histogram.percentile(95.0).unwrap_or(p50);
            output.insert(
                stage.clone(),
                StageStats {
                    p50,
                    p95,
                    window_size: histogram.samples.len(),
                    total_count: histogram.total_count,
                },
            );
        }
        output
    }
}

impl Default for StageMetricsAggregator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(value: u64) -> Duration {
        Duration::from_millis(value)
    }

    #[test]
    fn empty_aggregator_returns_none_for_percentile() {
        let agg = StageMetricsAggregator::new();
        assert!(agg.percentile("stt", 50.0).is_none());
    }

    #[test]
    fn empty_aggregator_returns_empty_snapshot() {
        let agg = StageMetricsAggregator::new();
        assert!(agg.snapshot().is_empty());
    }

    #[test]
    fn single_sample_returns_that_value_for_any_percentile() {
        let agg = StageMetricsAggregator::new();
        agg.record("stt", ms(150));
        assert_eq!(agg.percentile("stt", 50.0), Some(ms(150)));
        assert_eq!(agg.percentile("stt", 95.0), Some(ms(150)));
        assert_eq!(agg.percentile("stt", 0.0), Some(ms(150)));
        assert_eq!(agg.percentile("stt", 100.0), Some(ms(150)));
    }

    #[test]
    fn p50_for_uniform_samples_is_middle_value() {
        let agg = StageMetricsAggregator::new();
        for value_ms in 100..=200u64 {
            agg.record("stt", ms(value_ms));
        }
        let p50 = agg.percentile("stt", 50.0).unwrap();
        assert!(p50 >= ms(149) && p50 <= ms(151), "p50 = {:?}", p50);
    }

    #[test]
    fn p95_is_close_to_top_of_distribution() {
        let agg = StageMetricsAggregator::new();
        for value_ms in 100..=200u64 {
            agg.record("stt", ms(value_ms));
        }
        let p95 = agg.percentile("stt", 95.0).unwrap();
        assert!(p95 >= ms(190) && p95 <= ms(200), "p95 = {:?}", p95);
    }

    #[test]
    fn capacity_evicts_oldest_samples_first() {
        let agg = StageMetricsAggregator::with_capacity(3);
        agg.record("stt", ms(100));
        agg.record("stt", ms(200));
        agg.record("stt", ms(300));
        // Adding a 4th evicts the 100
        agg.record("stt", ms(400));
        let p50 = agg.percentile("stt", 50.0).unwrap();
        // Window now contains [200, 300, 400] — median is 300.
        assert_eq!(p50, ms(300));
    }

    #[test]
    fn total_count_survives_window_eviction() {
        let agg = StageMetricsAggregator::with_capacity(2);
        for _ in 0..10 {
            agg.record("stt", ms(100));
        }
        let stats = agg.snapshot();
        let stt = stats.get("stt").unwrap();
        assert_eq!(stt.total_count, 10);
        assert_eq!(stt.window_size, 2);
    }

    #[test]
    fn multiple_stages_are_tracked_independently() {
        let agg = StageMetricsAggregator::new();
        agg.record("stt", ms(150));
        agg.record("translate", ms(50));
        agg.record("tts", ms(300));

        let snap = agg.snapshot();
        assert_eq!(snap.len(), 3);
        assert_eq!(snap["stt"].p50, ms(150));
        assert_eq!(snap["translate"].p50, ms(50));
        assert_eq!(snap["tts"].p50, ms(300));
    }

    #[test]
    fn snapshot_omits_stages_with_no_samples() {
        let agg = StageMetricsAggregator::new();
        // No record() call → snapshot empty
        assert!(agg.snapshot().is_empty());
        agg.record("stt", ms(100));
        assert!(agg.snapshot().contains_key("stt"));
    }

    #[test]
    fn capacity_zero_is_clamped_to_one() {
        let agg = StageMetricsAggregator::with_capacity(0);
        agg.record("stt", ms(50));
        agg.record("stt", ms(100));
        // Window holds only the newest sample
        assert_eq!(agg.percentile("stt", 50.0), Some(ms(100)));
    }

    #[test]
    fn percentile_clamps_out_of_range_values() {
        let agg = StageMetricsAggregator::new();
        for value_ms in [10, 20, 30].iter() {
            agg.record("stt", ms(*value_ms));
        }
        // p > 100 clamps to 100, returning the max
        assert_eq!(agg.percentile("stt", 200.0), Some(ms(30)));
        // p < 0 clamps to 0, returning the min
        assert_eq!(agg.percentile("stt", -10.0), Some(ms(10)));
    }
}
