use shared::PipelineConfig;

fn main() {
    tracing_subscriber::fmt::init();

    let config = PipelineConfig::default();
    tracing::info!(
        "Meeting Translator starting — {} -> {}",
        config.speaker_source_language.display_name(),
        config.speaker_target_language.display_name()
    );

    tracing::info!("Scaffold complete. Pipeline stages not yet implemented.");
}
