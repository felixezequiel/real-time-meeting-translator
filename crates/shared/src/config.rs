use serde::{Deserialize, Serialize};

use crate::language::Language;

const DEFAULT_CHUNK_DURATION_MS: u64 = 2000;
const DEFAULT_WHISPER_MODEL: &str = "base";
const DEFAULT_TTS_SPEED: f32 = 1.1;

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
        }
    }
}

impl PipelineConfig {
    pub fn chunk_duration_seconds(&self) -> f32 {
        self.chunk_duration_ms as f32 / 1000.0
    }

    /// Returns the virtual mic device name, defaulting to "CABLE Input".
    pub fn effective_virtual_mic(&self) -> &str {
        self.virtual_mic_device.as_deref().unwrap_or("CABLE Input")
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
    fn default_chunk_duration_is_two_seconds() {
        let config = PipelineConfig::default();
        assert_eq!(config.chunk_duration_ms, 2000);
        assert_eq!(config.chunk_duration_seconds(), 2.0);
    }

    #[test]
    fn default_whisper_model_is_base() {
        let config = PipelineConfig::default();
        assert_eq!(config.whisper_model, "base");
    }

    #[test]
    fn effective_virtual_mic_defaults_to_cable_input() {
        let config = PipelineConfig::default();
        assert_eq!(config.effective_virtual_mic(), "CABLE Input");
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
