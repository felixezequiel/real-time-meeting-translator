use serde::{Deserialize, Serialize};

use crate::language::Language;

const DEFAULT_CHUNK_DURATION_MS: u64 = 2000;
const DEFAULT_WHISPER_MODEL: &str = "base";
const DEFAULT_TTS_SPEED: f32 = 1.1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub speaker_source_language: Language,
    pub speaker_target_language: Language,
    pub chunk_duration_ms: u64,
    pub whisper_model: String,
    pub tts_speed: f32,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            speaker_source_language: Language::English,
            speaker_target_language: Language::Portuguese,
            chunk_duration_ms: DEFAULT_CHUNK_DURATION_MS,
            whisper_model: DEFAULT_WHISPER_MODEL.to_string(),
            tts_speed: DEFAULT_TTS_SPEED,
        }
    }
}

impl PipelineConfig {
    pub fn chunk_duration_seconds(&self) -> f32 {
        self.chunk_duration_ms as f32 / 1000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_translates_english_to_portuguese() {
        let config = PipelineConfig::default();

        assert_eq!(config.speaker_source_language, Language::English);
        assert_eq!(config.speaker_target_language, Language::Portuguese);
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
    fn default_tts_speed_is_slightly_faster() {
        let config = PipelineConfig::default();
        assert_eq!(config.tts_speed, 1.1);
    }

    #[test]
    fn config_serializes_to_toml() {
        let config = PipelineConfig::default();
        let toml_string = toml::to_string(&config).expect("should serialize to TOML");

        assert!(toml_string.contains("speaker_source_language"));
        assert!(toml_string.contains("chunk_duration_ms"));
    }

    #[test]
    fn config_deserializes_from_toml() {
        let toml_input = r#"
            speaker_source_language = "English"
            speaker_target_language = "Portuguese"
            chunk_duration_ms = 3000
            whisper_model = "tiny"
            tts_speed = 1.0
        "#;

        let config: PipelineConfig =
            toml::from_str(toml_input).expect("should deserialize from TOML");

        assert_eq!(config.chunk_duration_ms, 3000);
        assert_eq!(config.whisper_model, "tiny");
    }
}
