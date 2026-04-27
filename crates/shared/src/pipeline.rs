use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::language::Language;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextSegment {
    pub text: String,
    pub language: Language,
    /// Identifier of the original speaker as assigned by online diarization.
    /// `None` means diarization is disabled or the speaker is still being
    /// bootstrapped; in that case the TTS layer falls back to a built-in voice.
    #[serde(default)]
    pub speaker_id: Option<u32>,
}

impl TextSegment {
    pub fn new(text: String, language: Language) -> Self {
        Self {
            text,
            language,
            speaker_id: None,
        }
    }

    /// Builder-style setter for the speaker id. Keeps existing call sites
    /// (`TextSegment::new(...)`) untouched when diarization is not wired yet.
    pub fn with_speaker_id(mut self, speaker_id: u32) -> Self {
        self.speaker_id = Some(speaker_id);
        self
    }

    pub fn is_empty(&self) -> bool {
        self.text.trim().is_empty()
    }

    pub fn word_count(&self) -> usize {
        self.text.split_whitespace().count()
    }
}

#[derive(Debug, Error)]
pub enum StageError {
    #[error("Stage processing failed: {0}")]
    ProcessingFailed(String),

    #[error("Stage not initialized: {0}")]
    NotInitialized(String),

    #[error("Stage timeout after {0:?}")]
    Timeout(Duration),
}

#[async_trait]
pub trait PipelineStage: Send + Sync {
    fn name(&self) -> &str;

    async fn initialize(&mut self) -> Result<(), StageError>;

    async fn shutdown(&mut self) -> Result<(), StageError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineCommand {
    Start,
    Stop,
}

#[derive(Debug, Clone)]
pub struct PipelineMetrics {
    pub stage_name: String,
    pub processing_duration: Duration,
}

impl PipelineMetrics {
    pub fn new(stage_name: String, processing_duration: Duration) -> Self {
        Self {
            stage_name,
            processing_duration,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_segment_is_empty_for_whitespace_only() {
        let segment = TextSegment::new("   ".to_string(), Language::English);
        assert!(segment.is_empty());
    }

    #[test]
    fn text_segment_is_not_empty_for_real_text() {
        let segment = TextSegment::new("hello world".to_string(), Language::English);
        assert!(!segment.is_empty());
    }

    #[test]
    fn text_segment_word_count_correct() {
        let segment = TextSegment::new("the quick brown fox".to_string(), Language::English);
        assert_eq!(segment.word_count(), 4);
    }

    #[test]
    fn text_segment_word_count_zero_for_empty() {
        let segment = TextSegment::new("".to_string(), Language::Portuguese);
        assert_eq!(segment.word_count(), 0);
    }

    #[test]
    fn pipeline_metrics_stores_duration() {
        let duration = Duration::from_millis(150);
        let metrics = PipelineMetrics::new("stt".to_string(), duration);

        assert_eq!(metrics.stage_name, "stt");
        assert_eq!(metrics.processing_duration, duration);
    }

    #[test]
    fn pipeline_command_equality() {
        assert_eq!(PipelineCommand::Start, PipelineCommand::Start);
        assert_ne!(PipelineCommand::Start, PipelineCommand::Stop);
    }

    #[test]
    fn stage_error_display_messages() {
        let error = StageError::ProcessingFailed("whisper crashed".to_string());
        assert_eq!(error.to_string(), "Stage processing failed: whisper crashed");

        let error = StageError::NotInitialized("tts".to_string());
        assert_eq!(error.to_string(), "Stage not initialized: tts");

        let error = StageError::Timeout(Duration::from_secs(5));
        assert_eq!(error.to_string(), "Stage timeout after 5s");
    }
}
