use serde::{Deserialize, Serialize};

use crate::language::Language;

/// Audio capture chunk size. With Silero VAD on the front of the pipeline
/// (ADR 0008), most chunks become a fast no-op when nothing is being said,
/// so we can afford a tighter cadence. 280 ms is small enough that the
/// boundary detector in `StreamingSession` reacts quickly to a finished
/// utterance and large enough that the diarisation bridge still has a
/// few syllables of content for an embedding. Values <250 ms cause the
/// streaming STT throttle (`MIN_INFERENCE_INTERVAL_MS = 250`) to drop
/// every other chunk before it reaches Whisper.
const DEFAULT_CHUNK_DURATION_MS: u64 = 280;
/// Small quantized (q5_1) — ~181 MB vs 466 MB fp16, ~2x faster inference
/// on both CPU and GPU with marginal quality loss for PT/EN.
const DEFAULT_WHISPER_MODEL: &str = "small-q5_1";
const DEFAULT_TTS_SPEED: f32 = 1.1;

fn default_enable_voice_conversion() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Language the other person speaks (what the speaker pipeline transcribes)
    pub speaker_source_language: Language,
    /// Language you want to hear (speaker pipeline TTS output)
    pub speaker_target_language: Language,

    /// Language you speak (what the mic pipeline transcribes)
    pub mic_source_language: Language,
    /// Language the other person should hear (mic pipeline TTS output)
    pub mic_target_language: Language,

    pub chunk_duration_ms: u64,
    pub whisper_model: String,
    pub tts_speed: f32,

    /// Output device name to capture via loopback (the speaker/headphone where the meeting plays).
    /// Accepts old key "input_device" for backward compatibility.
    #[serde(default, alias = "input_device")]
    pub loopback_device: Option<String>,

    /// Output device where you hear the translated speaker audio (your headphones).
    #[serde(default)]
    pub headphones_device: Option<String>,

    /// Physical microphone device name (your voice input).
    #[serde(default)]
    pub mic_device: Option<String>,

    /// Virtual microphone device the meeting app reads from (e.g. "CABLE Input").
    #[serde(default)]
    pub virtual_mic_device: Option<String>,

    /// Enable Sepformer source separation on the speaker pipeline. When
    /// true, each loopback audio chunk is split into two channels (one
    /// per simultaneous speaker) and processed by parallel STT branches
    /// — useful when meeting interruptions matter and the listener
    /// can't afford a "salad" transcript of overlapping voices. Costs
    /// ~50–80 ms per chunk + ~120 MB GPU/RAM. Default: false.
    #[serde(default)]
    pub enable_separation: bool,

    /// Absolute path to the user's recorded voice reference WAV.
    /// When set, the mic pipeline pre-extracts the speaker embedding
    /// from this file at startup and uses it for every TTS fragment —
    /// the auto-enrolment loop (which needs ~6 s of clean speech and
    /// only kicks in once per session) is skipped entirely on the
    /// mic side. Recording is driven by the settings UI ("Gravar
    /// minha voz" button) and the file lives under `<app_dir>/
    /// voice_profile/user.wav`. None means "no profile yet — fall
    /// back to the default Kokoro voice for outgoing translations".
    #[serde(default)]
    pub mic_voice_profile_path: Option<String>,

    /// Enable OpenVoice v2 tone-color conversion on the TTS output
    /// (ADR 0011). When true, after Kokoro synthesises a fragment the
    /// pipeline rewrites its timbre to match the actual speaker's
    /// voice (extracted from a 6-second reference auto-enrolled from
    /// live audio). Costs ~150–250 ms / fragment on GPU and ~50 MB
    /// for the TCC checkpoint. Falls back to raw Kokoro output if
    /// the bridge fails to start. Default: true.
    #[serde(default = "default_enable_voice_conversion")]
    pub enable_voice_conversion: bool,

    // NOTE: Earlier versions had `speaker_voice_reference_wav` and
    // `mic_voice_reference_wav` fields used by CosyVoice's zero-shot
    // cloning path. Those are gone now — voice differentiation comes
    // from per-speaker F0 tracking driving Piper's pitch shifter (no
    // pre-recorded references needed). Old config.toml files with those
    // keys still parse because we no longer reject unknown fields.
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            speaker_source_language: Language::English,
            speaker_target_language: Language::Portuguese,
            mic_source_language: Language::Portuguese,
            mic_target_language: Language::English,
            chunk_duration_ms: DEFAULT_CHUNK_DURATION_MS,
            whisper_model: DEFAULT_WHISPER_MODEL.to_string(),
            tts_speed: DEFAULT_TTS_SPEED,
            loopback_device: None,
            headphones_device: None,
            mic_device: None,
            virtual_mic_device: None,
            enable_separation: false,
            mic_voice_profile_path: None,
            enable_voice_conversion: true,
        }
    }
}

impl PipelineConfig {
    pub fn chunk_duration_seconds(&self) -> f32 {
        self.chunk_duration_ms as f32 / 1000.0
    }

    /// Returns the virtual mic device name, defaulting to "Hi-Fi Cable".
    /// Uses Hi-Fi Cable (separate from VB-Cable used for loopback) to avoid
    /// audio contamination between speaker and mic pipelines.
    /// Substring match: "Hi-Fi Cable" matches "Alto-falantes (VB-Audio Hi-Fi Cable)".
    pub fn effective_virtual_mic(&self) -> &str {
        self.virtual_mic_device.as_deref().unwrap_or("Hi-Fi Cable")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_speaker_pipeline_translates_english_to_portuguese() {
        let config = PipelineConfig::default();
        assert_eq!(config.speaker_source_language, Language::English);
        assert_eq!(config.speaker_target_language, Language::Portuguese);
    }

    #[test]
    fn default_mic_pipeline_translates_portuguese_to_english() {
        let config = PipelineConfig::default();
        assert_eq!(config.mic_source_language, Language::Portuguese);
        assert_eq!(config.mic_target_language, Language::English);
    }

    #[test]
    fn default_chunk_duration_matches_streaming_stt_cadence() {
        let config = PipelineConfig::default();
        assert_eq!(config.chunk_duration_ms, 280);
        assert!((config.chunk_duration_seconds() - 0.28).abs() < 1e-6);
    }

    #[test]
    fn default_whisper_model_is_small_quantized() {
        let config = PipelineConfig::default();
        assert_eq!(config.whisper_model, "small-q5_1");
    }

    #[test]
    fn effective_virtual_mic_defaults_to_hifi_cable() {
        let config = PipelineConfig::default();
        assert_eq!(config.effective_virtual_mic(), "Hi-Fi Cable");
    }

    #[test]
    fn effective_virtual_mic_uses_configured_value() {
        let mut config = PipelineConfig::default();
        config.virtual_mic_device = Some("My Virtual Mic".to_string());
        assert_eq!(config.effective_virtual_mic(), "My Virtual Mic");
    }

    #[test]
    fn config_serializes_to_toml() {
        let config = PipelineConfig::default();
        let toml_string = toml::to_string(&config).expect("should serialize to TOML");
        assert!(toml_string.contains("speaker_source_language"));
        assert!(toml_string.contains("mic_source_language"));
    }

    #[test]
    fn config_deserializes_old_input_device_alias() {
        let toml_input = r#"
            speaker_source_language = "English"
            speaker_target_language = "Portuguese"
            mic_source_language = "Portuguese"
            mic_target_language = "English"
            chunk_duration_ms = 2000
            whisper_model = "base"
            tts_speed = 1.1
            input_device = "Headphones"
        "#;
        let config: PipelineConfig = toml::from_str(toml_input).unwrap();
        assert_eq!(config.loopback_device.as_deref(), Some("Headphones"));
    }
}
