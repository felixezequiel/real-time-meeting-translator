use anyhow::Result;
use shared::{PipelineCommand, PipelineConfig, TranslationDirection};
use tokio::sync::mpsc;

use audio::capture::AudioCapture;
use audio::device;
use audio::playback::AudioPlayback;
use audio::vad::EnergyVad;
use pipeline::SpeakerPipeline;
use stt::WhisperStt;
use translation::OpusMtTranslator;
use tts::PiperTts;
use ui::{TrayAction, TrayUi};

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("meeting_translator=info,pipeline=info,stt=info,translation=info,tts=info,audio=info")
        .init();

    let config = load_config()?;
    tracing::info!(
        "Meeting Translator — {} -> {}",
        config.speaker_source_language.display_name(),
        config.speaker_target_language.display_name()
    );

    let rt = tokio::runtime::Runtime::new()?;

    rt.block_on(async {
        run_application(config).await
    })
}

async fn run_application(config: PipelineConfig) -> Result<()> {
    let project_dir = std::env::current_dir()?;
    let scripts_dir = project_dir.join("scripts");

    let stt_bridge_path = scripts_dir.join("stt_bridge.py");
    let translation_bridge_path = scripts_dir.join("translation_bridge.py");
    let tts_bridge_path = scripts_dir.join("tts_bridge.py");

    let direction = TranslationDirection::new(
        config.speaker_source_language,
        config.speaker_target_language,
    );

    // Use "base.en" as default model name for faster-whisper (downloads automatically)
    let whisper_model_name = config.whisper_model.clone();

    let mut stt = WhisperStt::new(
        stt_bridge_path,
        whisper_model_name.into(),
        config.speaker_source_language,
    );
    let mut translator = OpusMtTranslator::new(translation_bridge_path, direction);
    let mut tts_engine = PiperTts::new(tts_bridge_path, config.speaker_target_language);
    let vad = EnergyVad::with_defaults();

    tracing::info!("Initializing STT...");
    shared::PipelineStage::initialize(&mut stt).await?;

    tracing::info!("Initializing Translation bridge...");
    shared::PipelineStage::initialize(&mut translator).await?;

    tracing::info!("Initializing TTS bridge...");
    shared::PipelineStage::initialize(&mut tts_engine).await?;

    let audio_buffer_size = 16;
    let (capture_tx, capture_rx) = mpsc::channel(audio_buffer_size);
    let (playback_tx, playback_rx) = mpsc::channel(audio_buffer_size);
    let (command_tx, command_rx) = mpsc::channel(8);
    let (metrics_tx, mut metrics_rx) = mpsc::channel(64);

    let input_device = device::get_default_input_device()
        .map_err(|e| anyhow::anyhow!("No input device: {}", e))?;
    let input_name = cpal::traits::DeviceTrait::name(&input_device)
        .unwrap_or_else(|_| "Unknown".to_string());
    tracing::info!("Capture device: {}", input_name);

    let capture = AudioCapture::new(input_device, config.chunk_duration_ms);
    let _capture_stream = capture
        .start(capture_tx)
        .map_err(|e| anyhow::anyhow!("Capture failed: {}", e))?;

    let output_device = device::find_output_device_by_name("CABLE Input")
        .or_else(|_| {
            tracing::warn!("VB-Cable not found, using default output device");
            device::get_default_output_device()
        })
        .map_err(|e| anyhow::anyhow!("No output device: {}", e))?;
    let output_name = cpal::traits::DeviceTrait::name(&output_device)
        .unwrap_or_else(|_| "Unknown".to_string());
    tracing::info!("Playback device: {}", output_name);

    let playback = AudioPlayback::new(output_device);
    let _playback_stream = playback
        .start(playback_rx)
        .map_err(|e| anyhow::anyhow!("Playback failed: {}", e))?;

    let speaker_pipeline = SpeakerPipeline::new(stt, translator, tts_engine, vad);

    tokio::spawn(async move {
        speaker_pipeline
            .run(capture_rx, playback_tx, command_rx, metrics_tx)
            .await;
    });

    tokio::spawn(async move {
        while let Some(metric) = metrics_rx.recv().await {
            if metric.stage_name == "total" {
                tracing::info!(
                    "Pipeline latency: {}ms",
                    metric.processing_duration.as_millis()
                );
            }
        }
    });

    let mut tray = TrayUi::new().map_err(|e| anyhow::anyhow!("Tray UI failed: {}", e))?;

    let _ = command_tx.send(PipelineCommand::Start).await;
    tracing::info!("Pipeline auto-started. Use system tray to control.");

    loop {
        if let Some(action) = tray.process_events() {
            match action {
                TrayAction::Command(cmd) => {
                    let _ = command_tx.send(cmd).await;
                }
                TrayAction::Quit => {
                    tracing::info!("Shutting down...");
                    break;
                }
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }

    tracing::info!("Meeting Translator stopped");
    Ok(())
}

fn load_config() -> Result<PipelineConfig> {
    let config_path = std::env::current_dir()?.join("config.toml");
    if config_path.exists() {
        let content = std::fs::read_to_string(&config_path)?;
        let config: PipelineConfig = toml::from_str(&content)?;
        tracing::info!("Config loaded from {}", config_path.display());
        Ok(config)
    } else {
        tracing::info!("No config.toml found, using defaults");
        Ok(PipelineConfig::default())
    }
}
