//! Speaker pipeline V2 — adaptive sliding-window per ADR 0013.
//!
//! Replaces the streaming local-agreement path (ADR 0004) with a
//! phrase-aligned pipeline:
//!
//! ```text
//! [audio_input] → [VAD] → [PhraseSegmenter] → [STT batch] →
//!     [is_echo? translate? TTS? TCC?] → [audio_output]
//! ```
//!
//! Each window is a complete utterance (closed by silence or by the
//! adaptive max). Whisper, the translator and TTS receive a coherent
//! phrase instead of fragments — which fixes the "translation
//! horrível" problem and removes the need for streaming local-
//! agreement, accumulator state, or punctuation-driven flushes.
//!
//! V2 starts simple: it does **not** do per-speaker F0 tracking or
//! auto-enrolment of voice references. Those are Phase 2 work tracked
//! in ADR 0013. The Mic side (which already had a fixed voice
//! profile) keeps full TCC support; the loopback side uses raw TTS
//! until per-speaker enrolment lands.

use audio::phrase_segmenter::{PhraseSegmenter, PhraseSegmenterConfig};
use audio::SileroVad;
use diarization::OnlineDiarizer;
use shared::{AudioChunk, Language, PipelineCommand, PipelineMetrics};
use stt::WhisperStt;
use tokio::sync::mpsc;
use tracing;
use translation::OpusMtTranslator;
use tts::{PiperTts, VoiceProfile};
use voice_convert::ToneColorConverter;

use std::sync::Arc;
use std::sync::mpsc as std_mpsc;
use std::time::{Duration, Instant};

use crate::{is_echo, is_translation_degenerate, record_translation, EchoBuffer, VoiceProfileRegistry};

const WHISPER_SAMPLE_RATE: u32 = 16_000;

/// Energy floor used when Silero VAD is unavailable. Mirrors the
/// fallback in ADR 0008 / the legacy pipeline so behaviour stays
/// consistent on a fresh checkout.
const FALLBACK_MIN_RMS_FOR_STT: f32 = 0.003;

#[derive(Debug, Clone, Copy)]
pub struct V2Config {
    pub max_window: Duration,
    pub silence_tail: Duration,
    pub min_window: Duration,
}

impl Default for V2Config {
    fn default() -> Self {
        Self {
            max_window: Duration::from_millis(5000),
            silence_tail: Duration::from_millis(400),
            min_window: Duration::from_millis(600),
        }
    }
}

impl From<V2Config> for PhraseSegmenterConfig {
    fn from(c: V2Config) -> Self {
        PhraseSegmenterConfig {
            max_window: c.max_window,
            silence_tail: c.silence_tail,
            min_window: c.min_window,
        }
    }
}

pub struct SpeakerPipelineV2 {
    pub name: String,
    pub stt: Arc<WhisperStt>,
    pub translator: Arc<OpusMtTranslator>,
    pub tts: Arc<PiperTts>,
    pub source_language: Language,
    pub echo_buffer: EchoBuffer,
    pub vad: Option<Arc<SileroVad>>,
    pub diarizer: Option<Arc<OnlineDiarizer>>,
    pub voice_convert: Option<Arc<ToneColorConverter>>,
    pub fixed_voice_reference: Option<String>,
    pub config: V2Config,
    /// Optional channel for subtitle text events. When present, every
    /// translated phrase is forwarded to the overlay UI before TTS.
    /// Uses `std::sync::mpsc` because the send happens inside
    /// `spawn_blocking` (no async context).
    pub subtitle_tx: Option<std_mpsc::Sender<SubtitleEvent>>,
}

#[derive(Debug, Clone)]
pub struct SubtitleEvent {
    pub pipeline_name: String,
    pub source_text: String,
    pub translated_text: String,
    pub language: Language,
    pub timestamp: Instant,
}

impl SpeakerPipelineV2 {
    pub fn new(
        name: impl Into<String>,
        stt: Arc<WhisperStt>,
        translator: Arc<OpusMtTranslator>,
        tts: Arc<PiperTts>,
        source_language: Language,
        echo_buffer: EchoBuffer,
    ) -> Self {
        Self {
            name: name.into(),
            stt,
            translator,
            tts,
            source_language,
            echo_buffer,
            vad: None,
            diarizer: None,
            voice_convert: None,
            fixed_voice_reference: None,
            config: V2Config::default(),
            subtitle_tx: None,
        }
    }

    pub fn with_vad(mut self, vad: Arc<SileroVad>) -> Self {
        self.vad = Some(vad);
        self
    }

    pub fn with_diarizer(mut self, diarizer: Arc<OnlineDiarizer>) -> Self {
        self.diarizer = Some(diarizer);
        self
    }

    pub fn with_voice_convert(mut self, vc: Arc<ToneColorConverter>) -> Self {
        self.voice_convert = Some(vc);
        self
    }

    pub fn with_fixed_voice_reference(mut self, path: impl Into<String>) -> Self {
        self.fixed_voice_reference = Some(path.into());
        self
    }

    pub fn with_subtitle_channel(
        mut self,
        tx: std_mpsc::Sender<SubtitleEvent>,
    ) -> Self {
        self.subtitle_tx = Some(tx);
        self
    }

    pub fn with_config(mut self, config: V2Config) -> Self {
        self.config = config;
        self
    }

    pub async fn run(
        self,
        mut audio_input: mpsc::UnboundedReceiver<AudioChunk>,
        audio_output: mpsc::UnboundedSender<AudioChunk>,
        mut command_rx: mpsc::Receiver<PipelineCommand>,
        metrics_tx: mpsc::Sender<PipelineMetrics>,
    ) {
        let pipeline_name = self.name;
        let mut segmenter = PhraseSegmenter::new(
            WHISPER_SAMPLE_RATE,
            self.config.into(),
        );
        let min_speech_samples = segmenter.min_window_samples();
        // Voice profile registry: collects ~6 s of clean audio per
        // diarised speaker, writes a reference WAV, and exposes its
        // path so OpenVoice TCC can convert subsequent TTS output to
        // that speaker's timbre. Without diariser this stays empty
        // and the pipeline falls back to raw Kokoro output.
        let voice_profiles = Arc::new(VoiceProfileRegistry::new(
            pipeline_name.clone(),
            WHISPER_SAMPLE_RATE,
        ));
        let mut is_running = false;

        tracing::info!(
            "[{}] V2 ready (max_window={:?}, silence_tail={:?}, min_window={:?}, diariser={}, vc={})",
            pipeline_name,
            self.config.max_window,
            self.config.silence_tail,
            self.config.min_window,
            self.diarizer.is_some(),
            self.voice_convert.is_some(),
        );

        loop {
            tokio::select! {
                cmd = command_rx.recv() => {
                    match cmd {
                        Some(PipelineCommand::Start) => {
                            tracing::info!(
                                "[{}] V2 pipeline started (source={})",
                                pipeline_name,
                                self.source_language.display_name(),
                            );
                            is_running = true;
                            if let Some(v) = self.vad.as_ref() {
                                v.reset_state();
                            }
                            let _ = segmenter.flush();
                        }
                        Some(PipelineCommand::Stop) => {
                            tracing::info!("[{}] V2 pipeline stopped", pipeline_name);
                            is_running = false;
                            let _ = segmenter.flush();
                        }
                        None => return,
                    }
                }
                chunk = audio_input.recv() => {
                    let chunk = match chunk {
                        Some(c) => c,
                        None => return,
                    };
                    if !is_running { continue; }

                    let vad_start = Instant::now();
                    let is_speech = match self.vad.as_ref() {
                        Some(v) => v.has_speech(&chunk.samples),
                        None => rms(&chunk.samples) >= FALLBACK_MIN_RMS_FOR_STT,
                    };
                    let _ = metrics_tx.try_send(PipelineMetrics::new(
                        "vad".to_string(),
                        vad_start.elapsed(),
                    ));

                    let segment = segmenter.ingest(&chunk.samples, is_speech);
                    if let Some(segment) = segment {
                        if segment.speech_samples < min_speech_samples {
                            tracing::debug!(
                                "[{}] dropping short window ({} speech samples < {} min)",
                                pipeline_name,
                                segment.speech_samples,
                                min_speech_samples,
                            );
                            continue;
                        }

                        let stt = Arc::clone(&self.stt);
                        let translator = Arc::clone(&self.translator);
                        let tts_engine = Arc::clone(&self.tts);
                        let voice_convert = self.voice_convert.clone();
                        let diarizer = self.diarizer.clone();
                        let voice_profiles = Arc::clone(&voice_profiles);
                        let echo_buffer = Arc::clone(&self.echo_buffer);
                        let audio_output = audio_output.clone();
                        let metrics_tx = metrics_tx.clone();
                        let pipeline_name = pipeline_name.clone();
                        let source_language = self.source_language;
                        let fixed_voice_reference = self.fixed_voice_reference.clone();
                        let subtitle_tx = self.subtitle_tx.clone();
                        let segment_samples = segment.samples;
                        let segment_sample_rate = segment.sample_rate;

                        tokio::task::spawn_blocking(move || {
                            process_segment(
                                pipeline_name,
                                segment_samples,
                                segment_sample_rate,
                                stt,
                                translator,
                                tts_engine,
                                voice_convert,
                                diarizer,
                                voice_profiles,
                                echo_buffer,
                                source_language,
                                fixed_voice_reference,
                                audio_output,
                                metrics_tx,
                                subtitle_tx,
                            );
                        });
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn process_segment(
    pipeline_name: String,
    samples: Vec<f32>,
    sample_rate: u32,
    stt: Arc<WhisperStt>,
    translator: Arc<OpusMtTranslator>,
    tts: Arc<PiperTts>,
    voice_convert: Option<Arc<ToneColorConverter>>,
    diarizer: Option<Arc<OnlineDiarizer>>,
    voice_profiles: Arc<VoiceProfileRegistry>,
    echo_buffer: EchoBuffer,
    source_language: Language,
    fixed_voice_reference: Option<String>,
    audio_output: mpsc::UnboundedSender<AudioChunk>,
    metrics_tx: mpsc::Sender<PipelineMetrics>,
    subtitle_tx: Option<std_mpsc::Sender<SubtitleEvent>>,
) {
    let total_start = Instant::now();

    // ─── Diariser ─────────────────────────────────────────────────────
    // Run BEFORE STT so we can route the right voice reference into TCC.
    // When fixed_voice_reference is set (mic side), skip diariser entirely
    // — only one person ever speaks into the mic, no need to identify.
    let diar_start = Instant::now();
    let identification = if fixed_voice_reference.is_some() {
        None
    } else {
        diarizer.as_ref().and_then(|d| match d.identify(&samples, sample_rate) {
            Ok(opt) => opt,
            Err(e) => {
                tracing::warn!("[{}] V2 diariser failed: {}", pipeline_name, e);
                None
            }
        })
    };
    let _ = metrics_tx.try_send(PipelineMetrics::new(
        "diarize".to_string(),
        diar_start.elapsed(),
    ));

    let speaker_id = identification.as_ref().map(|s| s.speaker_id);
    if let Some(id) = identification.as_ref() {
        voice_profiles.record_f0(id.speaker_id, id.f0_hz);
        // Auto-enrolment: feed the segment audio into this speaker's
        // reference buffer. Once it has 6 s of clean speech a WAV
        // reference is written and TCC can convert future TTS for the
        // same speaker.
        voice_profiles.ingest_audio(id.speaker_id, &samples);
    }

    let chunk = AudioChunk::new(samples, sample_rate, 1);

    // ─── STT ──────────────────────────────────────────────────────────
    let stt_start = Instant::now();
    let transcribed = match stt.transcribe(&chunk, source_language) {
        Ok(t) => t,
        Err(e) => {
            tracing::warn!("[{}] V2 STT failed: {}", pipeline_name, e);
            return;
        }
    };
    let _ = metrics_tx.try_send(PipelineMetrics::new(
        "stt".to_string(),
        stt_start.elapsed(),
    ));

    let source_text = transcribed.text.trim();
    if source_text.is_empty() {
        return;
    }

    if is_echo(source_text, &echo_buffer) {
        tracing::info!(
            "[{}] V2 echo dropped: \"{}\"",
            pipeline_name,
            &source_text[..source_text.len().min(60)],
        );
        return;
    }

    tracing::info!(
        "[{}] V2 STT: \"{}\"",
        pipeline_name,
        &source_text[..source_text.len().min(80)],
    );

    // ─── Translate ────────────────────────────────────────────────────
    let translate_start = Instant::now();
    let translated = match translator.translate(&transcribed) {
        Ok(t) => t,
        Err(e) => {
            tracing::warn!("[{}] V2 translation failed: {}", pipeline_name, e);
            return;
        }
    };
    let _ = metrics_tx.try_send(PipelineMetrics::new(
        "translate".to_string(),
        translate_start.elapsed(),
    ));

    let translated_text = translated.text.trim();
    if translated_text.is_empty() {
        return;
    }
    if is_translation_degenerate(source_text, translated_text) {
        tracing::info!(
            "[{}] V2 degenerate translation dropped: \"{}\" → \"{}\"",
            pipeline_name,
            &source_text[..source_text.len().min(40)],
            &translated_text[..translated_text.len().min(40)],
        );
        return;
    }

    record_translation(&echo_buffer, translated_text);

    if let Some(tx) = subtitle_tx.as_ref() {
        let _ = tx.send(SubtitleEvent {
            pipeline_name: pipeline_name.clone(),
            source_text: source_text.to_string(),
            translated_text: translated_text.to_string(),
            language: translated.language,
            timestamp: Instant::now(),
        });
    }

    tracing::info!(
        "[{}] V2 → \"{}\"",
        pipeline_name,
        &translated_text[..translated_text.len().min(80)],
    );

    // ─── TTS ──────────────────────────────────────────────────────────
    let tts_start = Instant::now();
    // VoiceProfile: when we have a speaker_id we feed the running-mean
    // F0 to Kokoro. The bridge picks a sticky voice per speaker (so the
    // same person keeps the same voice across utterances) and biases
    // pitch toward their measured F0. Without diariser, default profile.
    let voice_profile = match speaker_id {
        Some(id) => VoiceProfile {
            target_f0_hz: voice_profiles.f0_for(id),
            formant_shift: 1.0,
            speaker_id: Some(id),
        },
        None => VoiceProfile::default(),
    };
    let tts_audio = match tts.synthesize(&translated, voice_profile) {
        Ok(a) => a,
        Err(e) => {
            tracing::warn!("[{}] V2 TTS failed: {}", pipeline_name, e);
            return;
        }
    };
    let _ = metrics_tx.try_send(PipelineMetrics::new(
        "tts".to_string(),
        tts_start.elapsed(),
    ));

    // ─── Voice conversion ─────────────────────────────────────────────
    // Reference selection in order of preference:
    //   1. fixed_voice_reference (mic side: user's enrolled voice)
    //   2. per-speaker auto-enrolled WAV from VoiceProfileRegistry
    //   3. none → raw TTS output
    let speaker_for_tcc = speaker_id.unwrap_or(0);
    let reference_path = fixed_voice_reference
        .clone()
        .or_else(|| speaker_id.and_then(|id| voice_profiles.reference_for(id)));
    let final_audio = match (voice_convert.as_ref(), reference_path.as_deref()) {
        (Some(vc), Some(reference)) => {
            let vc_start = Instant::now();
            let converted = match vc.convert(&tts_audio, reference, speaker_for_tcc) {
                Ok(Some(c)) => Some(AudioChunk::new(c.samples, c.sample_rate, 1)),
                Ok(None) => None,
                Err(e) => {
                    tracing::warn!("[{}] V2 voice convert failed: {}", pipeline_name, e);
                    None
                }
            };
            let _ = metrics_tx.try_send(PipelineMetrics::new(
                "voice_convert".to_string(),
                vc_start.elapsed(),
            ));
            converted.unwrap_or(tts_audio)
        }
        _ => tts_audio,
    };

    let _ = audio_output.send(final_audio);
    let _ = metrics_tx.try_send(PipelineMetrics::new(
        "total".to_string(),
        total_start.elapsed(),
    ));
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}
