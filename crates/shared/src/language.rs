use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Language {
    Portuguese,
    English,
}

impl Language {
    pub fn whisper_code(&self) -> &'static str {
        match self {
            Language::Portuguese => "pt",
            Language::English => "en",
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            Language::Portuguese => "Português",
            Language::English => "English",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranslationDirection {
    pub source: Language,
    pub target: Language,
}

impl TranslationDirection {
    pub fn new(source: Language, target: Language) -> Self {
        Self { source, target }
    }

    pub fn reversed(&self) -> Self {
        Self {
            source: self.target,
            target: self.source,
        }
    }
}

impl std::fmt::Display for TranslationDirection {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{} -> {}",
            self.source.display_name(),
            self.target.display_name()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whisper_code_returns_correct_locale() {
        assert_eq!(Language::Portuguese.whisper_code(), "pt");
        assert_eq!(Language::English.whisper_code(), "en");
    }

    #[test]
    fn translation_direction_reversed_swaps_source_and_target() {
        let en_to_pt = TranslationDirection::new(Language::English, Language::Portuguese);
        let reversed = en_to_pt.reversed();

        assert_eq!(reversed.source, Language::Portuguese);
        assert_eq!(reversed.target, Language::English);
    }

    #[test]
    fn translation_direction_display_shows_readable_format() {
        let direction = TranslationDirection::new(Language::English, Language::Portuguese);
        assert_eq!(direction.to_string(), "English -> Português");
    }
}
