use async_trait::async_trait;
use shared::{AudioChunk, Language, PipelineStage, StageError, TextSegment};
use std::path::PathBuf;
use std::sync::Mutex;
use thiserror::Error;
use tracing;

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

#[derive(Debug, Error)]
pub enum SttError {
    #[error("STT not initialized")]
    NotInitialized,

    #[error("Transcription failed: {0}")]
    TranscriptionFailed(String),
}

/// Native Whisper STT using whisper-rs (whisper.cpp bindings).
///
/// No Python subprocess, no temp WAV files, no stdin/stdout serialization.
/// Audio is passed directly as f32 samples in memory.
pub struct WhisperStt {
    model_path: PathBuf,
    ctx: Option<Mutex<WhisperContext>>,
    language: Language,
}

impl WhisperStt {
    pub fn new(_bridge_script_path: PathBuf, model_path: PathBuf, language: Language) -> Self {
        // bridge_script_path is ignored — we no longer use the Python bridge.
        // The parameter is kept for API compatibility during migration.
        Self {
            model_path,
            ctx: None,
            language,
        }
    }

    pub fn transcribe(&self, chunk: &AudioChunk) -> Result<TextSegment, SttError> {
        let ctx_mutex = self.ctx.as_ref().ok_or(SttError::NotInitialized)?;
        let ctx = ctx_mutex
            .lock()
            .map_err(|e| SttError::TranscriptionFailed(format!("Lock poisoned: {}", e)))?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

        // Auto-detect language — do NOT force it.
        // This is critical for the feedback-loop guard: if TTS output leaks back
        // into the loopback, Whisper detects it as the target language and the
        // pipeline drops it.
        params.set_language(None);
        params.set_translate(false);
        params.set_no_timestamps(true);
        params.set_single_segment(false);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_suppress_blank(true);
        params.set_suppress_nst(true);
        // Temperature 0 = greedy decoding (fastest)
        params.set_temperature(0.0);
        // No speech detection threshold
        params.set_no_speech_thold(0.6);

        // Run inference — this is the only blocking call, no file I/O
        let mut state = ctx
            .create_state()
            .map_err(|e| SttError::TranscriptionFailed(format!("State creation failed: {}", e)))?;

        state
            .full(params, &chunk.samples)
            .map_err(|e| SttError::TranscriptionFailed(format!("Inference failed: {}", e)))?;

        // Collect all segments
        let num_segments = state.full_n_segments();

        let mut text_parts = Vec::new();
        for i in 0..num_segments {
            if let Some(segment) = state.get_segment(i) {
                if let Ok(text) = segment.to_str_lossy() {
                    let trimmed = text.trim();
                    if !trimmed.is_empty() {
                        text_parts.push(trimmed.to_string());
                    }
                }
            }
        }

        let full_text = text_parts.join(" ").trim().to_string();

        // Detect language using whisper.cpp's internal detection + text heuristic
        let lang_id = state.full_lang_id_from_state();
        let detected_lang = match whisper_rs::get_lang_str(lang_id) {
            Some("pt") => Language::Portuguese,
            Some("en") => Language::English,
            _ => detect_language_from_text_heuristic(&full_text, self.language),
        };

        // Check for repetitive hallucinations
        if is_repetitive(&full_text) {
            tracing::debug!("Hallucination detected, discarding: {}...", &full_text[..full_text.len().min(60)]);
            return Ok(TextSegment::new(String::new(), detected_lang));
        }

        Ok(TextSegment::new(full_text, detected_lang))
    }
}

/// Simple language detection heuristic based on common words.
/// whisper-rs doesn't expose the detected language directly through
/// the state API, so we use a word-frequency approach as fallback.
fn detect_language_from_text_heuristic(text: &str, expected: Language) -> Language {
    if text.is_empty() {
        return expected;
    }

    let lower = text.to_lowercase();

    // Portuguese indicators
    let pt_words = ["que", "não", "uma", "com", "para", "como", "mais", "foi",
                    "são", "tem", "está", "isso", "esse", "esta", "seu", "sua",
                    "por", "dos", "das", "nos", "nas", "aos", "às", "muito",
                    "também", "então", "ainda", "sobre", "depois"];

    // English indicators
    let en_words = ["the", "and", "was", "for", "that", "with", "this", "from",
                    "but", "not", "are", "were", "been", "have", "has", "had",
                    "will", "would", "could", "should", "their", "which", "there",
                    "when", "what", "about", "into", "than", "them"];

    let words: Vec<&str> = lower.split_whitespace().collect();
    if words.len() < 3 {
        return expected;
    }

    let pt_score: usize = words.iter().filter(|w| pt_words.contains(&w.trim_matches(|c: char| !c.is_alphabetic()))).count();
    let en_score: usize = words.iter().filter(|w| en_words.contains(&w.trim_matches(|c: char| !c.is_alphabetic()))).count();

    if pt_score > en_score && pt_score >= 2 {
        Language::Portuguese
    } else if en_score > pt_score && en_score >= 2 {
        Language::English
    } else {
        expected
    }
}

/// Detect repetitive hallucination patterns like "i'r un i'r un..."
fn is_repetitive(text: &str) -> bool {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < 6 {
        return false;
    }

    // Check if any short phrase repeats more than 4 times
    for pattern_len in 1..4 {
        let pattern: String = words[..pattern_len].join(" ");
        if pattern.len() < 2 { continue; }
        let count = text.matches(&pattern).count();
        if count > 4 { return true; }
    }

    // Check if unique words are less than 20% of total
    let unique: std::collections::HashSet<String> = words.iter()
        .map(|w| w.to_lowercase().trim_matches(|c: char| !c.is_alphabetic()).to_string())
        .collect();
    if words.len() > 10 && (unique.len() as f32 / words.len() as f32) < 0.2 {
        return true;
    }

    false
}

#[async_trait]
impl PipelineStage for WhisperStt {
    fn name(&self) -> &str {
        "whisper-stt-native"
    }

    async fn initialize(&mut self) -> Result<(), StageError> {
        let model_str = self.model_path.to_string_lossy().to_string();

        // Try to find the GGML model file
        let ggml_path = find_ggml_model(&model_str)
            .ok_or_else(|| StageError::NotInitialized(
                format!("GGML model not found for '{}'. Expected models/ggml-{}.bin", model_str, model_str)
            ))?;

        tracing::info!("Loading whisper.cpp model: {}", ggml_path.display());

        let params = WhisperContextParameters::default();
        let path_str = ggml_path.to_string_lossy().to_string();
        let ctx = WhisperContext::new_with_params(&path_str, params)
            .map_err(|e| StageError::NotInitialized(format!("Failed to load model: {}", e)))?;

        tracing::info!("whisper.cpp model loaded (native, no Python)");
        self.ctx = Some(Mutex::new(ctx));
        Ok(())
    }

    async fn shutdown(&mut self) -> Result<(), StageError> {
        self.ctx = None;
        tracing::info!("whisper.cpp model unloaded");
        Ok(())
    }
}

/// Find the GGML model file. Searches:
/// 1. models/ggml-{name}.bin (relative to working directory)
/// 2. The path as-is (if it's already a full path or "small" etc)
fn find_ggml_model(name: &str) -> Option<PathBuf> {
    // Try models/ggml-{name}.bin
    let ggml_path = PathBuf::from(format!("models/ggml-{}.bin", name));
    if ggml_path.exists() {
        return Some(ggml_path);
    }

    // Try the name as a direct path
    let direct = PathBuf::from(name);
    if direct.exists() {
        return Some(direct);
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_english_text() {
        let lang = detect_language_from_text_heuristic(
            "The piper was located in the north sea",
            Language::Portuguese,
        );
        assert_eq!(lang, Language::English);
    }

    #[test]
    fn detect_portuguese_text() {
        let lang = detect_language_from_text_heuristic(
            "O piper estava localizado no mar do norte para isso",
            Language::English,
        );
        assert_eq!(lang, Language::Portuguese);
    }

    #[test]
    fn short_text_returns_expected() {
        let lang = detect_language_from_text_heuristic("Hi", Language::Portuguese);
        assert_eq!(lang, Language::Portuguese);
    }

    #[test]
    fn repetitive_text_detected() {
        assert!(is_repetitive("un ir un ir un ir un ir un ir un ir"));
        assert!(!is_repetitive("The quick brown fox jumps over the lazy dog"));
    }
}
