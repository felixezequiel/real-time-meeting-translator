use async_trait::async_trait;
use shared::{AudioChunk, Language, PipelineStage, StageError, TextSegment};
use std::path::PathBuf;
use std::sync::Mutex;
use thiserror::Error;
use tracing;

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState};

pub mod streaming;
pub use streaming::{CommittedWords, StreamingSession};

#[derive(Debug, Error)]
pub enum SttError {
    #[error("STT not initialized")]
    NotInitialized,

    #[error("Transcription failed: {0}")]
    TranscriptionFailed(String),
}

/// Native Whisper STT using whisper-rs (whisper.cpp bindings).
///
/// The WhisperState is created ONCE during initialize() and reused for
/// every transcription — avoids re-allocating ~460MB of GPU compute
/// buffers per chunk.
pub struct WhisperStt {
    model_path: PathBuf,
    /// The state holds GPU memory and is reused across calls.
    /// The WhisperContext is kept alive inside the state via Arc.
    state: Option<Mutex<WhisperState>>,
}

impl WhisperStt {
    pub fn new(_bridge_script_path: PathBuf, model_path: PathBuf, _language: Language) -> Self {
        Self {
            model_path,
            state: None,
        }
    }

    /// Transcribe audio in the given language. The language is passed per-call
    /// so it can change at runtime when the user switches languages in the UI.
    pub fn transcribe(&self, chunk: &AudioChunk, language: Language) -> Result<TextSegment, SttError> {
        let state_mutex = self.state.as_ref().ok_or(SttError::NotInitialized)?;
        let mut state = state_mutex
            .lock()
            .map_err(|e| SttError::TranscriptionFailed(format!("Lock poisoned: {}", e)))?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

        // Force expected language — prevents Whisper from hallucinating in
        // the wrong language (e.g. generating Portuguese on English audio).
        params.set_language(Some(language.whisper_code()));
        params.set_translate(false);
        params.set_no_timestamps(true);
        params.set_single_segment(false);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_suppress_blank(true);
        params.set_suppress_nst(true);

        // Anti-hallucination
        params.set_temperature(0.0);
        params.set_no_speech_thold(0.5);
        params.set_n_max_text_ctx(128);
        params.set_entropy_thold(2.4);
        params.set_logprob_thold(-1.0);

        // Reuse state — no GPU re-initialization
        state
            .full(params, &chunk.samples)
            .map_err(|e| SttError::TranscriptionFailed(format!("Inference failed: {}", e)))?;

        // Collect segments
        let num_segments = state.full_n_segments();
        let mut text_parts = Vec::new();
        for i in 0..num_segments {
            if let Some(segment) = state.get_segment(i) {
                if segment.no_speech_probability() > 0.7 {
                    continue;
                }
                if let Ok(text) = segment.to_str_lossy() {
                    let trimmed = text.trim();
                    if !trimmed.is_empty() {
                        text_parts.push(trimmed.to_string());
                    }
                }
            }
        }

        let full_text = text_parts.join(" ").trim().to_string();

        // Language detection — only accept English or Portuguese.
        // If Whisper detects any other language (German, Welsh, etc.),
        // the audio is either garbage or a hallucination — drop it.
        let lang_id = state.full_lang_id_from_state();
        let detected_lang = match whisper_rs::get_lang_str(lang_id) {
            Some("pt") => Language::Portuguese,
            Some("en") => Language::English,
            Some(other) => {
                // Whisper detected a language other than en/pt.
                // Use text heuristic as fallback — if it recognizes en/pt, trust it.
                // Otherwise assume expected language instead of dropping the segment.
                let heuristic = detect_language_from_text_heuristic(&full_text, language);
                tracing::debug!("Whisper detected '{}', heuristic={:?} — text: \"{}\"",
                    other, heuristic, &full_text[..full_text.len().min(60)]);
                heuristic
            }
            None => detect_language_from_text_heuristic(&full_text, language),
        };

        if is_repetitive(&full_text) {
            return Ok(TextSegment::new(String::new(), detected_lang));
        }

        Ok(TextSegment::new(full_text, detected_lang))
    }
}

fn detect_language_from_text_heuristic(text: &str, expected: Language) -> Language {
    if text.is_empty() { return expected; }
    let lower = text.to_lowercase();

    let pt_words = ["que", "não", "uma", "com", "para", "como", "mais", "foi",
                    "são", "tem", "está", "isso", "esse", "esta", "seu", "sua",
                    "por", "dos", "das", "nos", "nas", "aos", "muito",
                    "também", "então", "ainda", "sobre", "depois"];
    let en_words = ["the", "and", "was", "for", "that", "with", "this", "from",
                    "but", "not", "are", "were", "been", "have", "has", "had",
                    "will", "would", "could", "should", "their", "which", "there",
                    "when", "what", "about", "into", "than", "them"];

    let words: Vec<&str> = lower.split_whitespace().collect();
    if words.len() < 3 { return expected; }

    let pt_score: usize = words.iter().filter(|w| pt_words.contains(&w.trim_matches(|c: char| !c.is_alphabetic()))).count();
    let en_score: usize = words.iter().filter(|w| en_words.contains(&w.trim_matches(|c: char| !c.is_alphabetic()))).count();

    if pt_score > en_score && pt_score >= 2 { Language::Portuguese }
    else if en_score > pt_score && en_score >= 2 { Language::English }
    else { expected }
}

fn is_repetitive(text: &str) -> bool {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < 4 { return false; }

    for pattern_len in 1..4.min(words.len()) {
        let pattern: String = words[..pattern_len].join(" ");
        if pattern.len() < 2 { continue; }
        if text.matches(&pattern).count() > 3 { return true; }
    }

    let unique: std::collections::HashSet<String> = words.iter()
        .map(|w| w.to_lowercase().trim_matches(|c: char| !c.is_alphabetic()).to_string())
        .filter(|w| !w.is_empty())
        .collect();
    if words.len() > 6 && (unique.len() as f32 / words.len() as f32) < 0.3 { return true; }

    let non_alpha = text.chars().filter(|c| !c.is_alphanumeric() && !c.is_whitespace() && !".,'!?-".contains(*c)).count() as f32;
    if non_alpha / text.len().max(1) as f32 > 0.3 { return true; }

    false
}

/// Check if CUDA is available by attempting to load ALL required NVIDIA DLLs.
/// This avoids calling any CUDA functions (which would trigger delay-loaded
/// DLL resolution and crash if CUDA Toolkit is not installed).
/// Requires: NVIDIA driver (nvcuda.dll) + CUDA Toolkit (cublas, cudart).
#[cfg(windows)]
fn is_cuda_available() -> bool {
    let required_dlls = [
        "nvcuda.dll",      // NVIDIA driver
        "cublas64_12.dll", // CUDA Toolkit
        "cudart64_12.dll", // CUDA Runtime
    ];

    for dll_name in required_dlls {
        let c_name = std::ffi::CString::new(dll_name).unwrap();
        let handle = unsafe {
            windows_sys::Win32::System::LibraryLoader::LoadLibraryA(c_name.as_ptr() as *const u8)
        };
        if handle.is_null() {
            tracing::debug!("CUDA DLL not found: {} — falling back to CPU", dll_name);
            return false;
        }
        unsafe { windows_sys::Win32::Foundation::FreeLibrary(handle); }
    }
    true
}

#[cfg(not(windows))]
fn is_cuda_available() -> bool {
    false
}

/// Find GGML model file.
/// Searches in order:
/// 1. `models/ggml-{name}.bin` relative to the executable directory (installed mode)
/// 2. `models/ggml-{name}.bin` relative to the current working directory (dev mode)
/// 3. Absolute/direct path (if `name` is already a full path)
fn find_ggml_model(name: &str) -> Option<PathBuf> {
    let ggml_filename = format!("models/ggml-{}.bin", name);

    // 1. Relative to exe directory (installed mode)
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            let path = exe_dir.join(&ggml_filename);
            if path.exists() { return Some(path); }
        }
    }
    // 2. Relative to current working directory (dev mode / cargo run)
    let path = PathBuf::from(&ggml_filename);
    if path.exists() { return Some(path); }
    // 3. Direct/absolute path
    let direct = PathBuf::from(name);
    if direct.exists() { return Some(direct); }
    None
}

#[async_trait]
impl PipelineStage for WhisperStt {
    fn name(&self) -> &str { "whisper-stt-native" }

    async fn initialize(&mut self) -> Result<(), StageError> {
        let model_str = self.model_path.to_string_lossy().to_string();
        let ggml_path = find_ggml_model(&model_str)
            .ok_or_else(|| StageError::NotInitialized(
                format!("GGML model not found for '{}'. Expected models/ggml-{}.bin", model_str, model_str)
            ))?;

        let gpu_available = is_cuda_available();
        tracing::info!(
            "Loading whisper.cpp model: {} (GPU: {})",
            ggml_path.display(),
            if gpu_available { "yes" } else { "no — using CPU" }
        );

        let mut params = WhisperContextParameters::default();
        params.use_gpu(gpu_available);

        let path_str = ggml_path.to_string_lossy().to_string();
        let ctx = WhisperContext::new_with_params(&path_str, params)
            .map_err(|e| StageError::NotInitialized(format!("Failed to load model: {}", e)))?;

        // Create state ONCE — reused for all transcriptions.
        // This avoids re-allocating ~460MB of GPU compute buffers per chunk.
        let state = ctx.create_state()
            .map_err(|e| StageError::NotInitialized(format!("Failed to create state: {}", e)))?;

        tracing::info!("whisper.cpp ready (state created, {} backend)", if gpu_available { "GPU" } else { "CPU" });
        self.state = Some(Mutex::new(state));
        Ok(())
    }

    async fn shutdown(&mut self) -> Result<(), StageError> {
        self.state = None;
        tracing::info!("whisper.cpp model unloaded");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_english_text() {
        let lang = detect_language_from_text_heuristic("The piper was located in the north sea", Language::Portuguese);
        assert_eq!(lang, Language::English);
    }

    #[test]
    fn detect_portuguese_text() {
        let lang = detect_language_from_text_heuristic("O piper estava localizado no mar do norte para isso", Language::English);
        assert_eq!(lang, Language::Portuguese);
    }

    #[test]
    fn short_text_returns_expected() {
        assert_eq!(detect_language_from_text_heuristic("Hi", Language::Portuguese), Language::Portuguese);
    }

    #[test]
    fn repetitive_text_detected() {
        assert!(is_repetitive("un ir un ir un ir un ir un ir un ir"));
        assert!(!is_repetitive("The quick brown fox jumps over the lazy dog"));
    }
}
