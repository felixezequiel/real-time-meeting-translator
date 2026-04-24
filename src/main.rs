use anyhow::Result;
use shared::{Language, PipelineCommand, PipelineConfig, PipelineStage, TranslationDirection};
use std::sync::Arc;
use tokio::sync::mpsc;

use audio::audio_switch;
use audio::capture::AudioCapture;
use audio::device;
use audio::loopback::LoopbackCapture;
use audio::playback::{AudioPlayback, MixerPlayback};
use pipeline::SpeakerPipeline;
use stt::WhisperStt;
use translation::OpusMtTranslator;
use tts::PiperTts;
use ui::{TrayAction, TrayUi};

/// VB-Cable A: used as system default output so meeting audio flows here.
/// Loopback captures from this device (meeting audio only).
/// Uses "VB-Audio Virtual Cable" (not "CABLE Input") to avoid matching
/// "Hi-Fi Cable Input (VB-Audio Hi-Fi Cable)" via substring search.
const VIRTUAL_CABLE_NAME: &str = "VB-Audio Virtual Cable";

/// Hi-Fi Cable: capture endpoint set as system default microphone.
/// Meeting apps read translated mic audio from this device.
const HIFI_CABLE_OUTPUT_NAME: &str = "Hi-Fi Cable";

#[cfg(windows)]
use windows_sys::Win32::UI::WindowsAndMessaging::{
    DispatchMessageW, PeekMessageW, TranslateMessage, MSG, PM_REMOVE,
};

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            "meeting_translator=info,pipeline=info,stt=warn,translation=warn,tts=warn,audio=warn",
        )
        .init();

    let config = load_config()?;
    tracing::info!(
        "Meeting Translator — speaker: {} → {}  |  mic: {} → {}",
        config.speaker_source_language.display_name(),
        config.speaker_target_language.display_name(),
        config.mic_source_language.display_name(),
        config.mic_target_language.display_name(),
    );

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async { run_application(config).await })
}

// ─── Loaded models (expensive — initialized once, never dropped) ──────────────

/// All model bridges, named by **direction** (not by pipeline).
/// This ensures `models_for_source` always returns the correct translator+TTS
/// regardless of what the startup config said.
struct LoadedModels {
    /// STT process A (auto-detects language — used by the speaker pipeline)
    stt_a: Arc<WhisperStt>,
    /// STT process B (auto-detects language — used by the mic pipeline)
    stt_b: Arc<WhisperStt>,
    /// English → Portuguese translator (always this direction)
    translator_en_pt: Arc<OpusMtTranslator>,
    /// Portuguese → English translator (always this direction)
    translator_pt_en: Arc<OpusMtTranslator>,
    /// Portuguese voice TTS (always Portuguese output)
    tts_portuguese: Arc<PiperTts>,
    /// English voice TTS (always English output)
    tts_english: Arc<PiperTts>,
}

// ─── Active audio pipelines (cheap — can be restarted on device change) ──────

struct ActivePipelines {
    /// Keeps the speaker loopback capture alive
    _speaker_capture: cpal::Stream,
    /// Mixer output: passthrough (raw meeting audio) + TTS in one stream,
    /// with smooth ducking applied to the passthrough while TTS speaks.
    _speaker_mixer: cpal::Stream,
    /// Keeps the mic capture alive
    _mic_capture: cpal::Stream,
    /// Keeps the mic TTS playback alive
    _mic_playback: cpal::Stream,
    /// Send Start/Stop to the speaker pipeline task
    speaker_cmd_tx: mpsc::Sender<PipelineCommand>,
    /// Send Start/Stop to the mic pipeline task
    mic_cmd_tx: mpsc::Sender<PipelineCommand>,
}

impl ActivePipelines {
    async fn send_command(&self, cmd: PipelineCommand) {
        let _ = self.speaker_cmd_tx.send(cmd).await;
        let _ = self.mic_cmd_tx.send(cmd).await;
    }
}

// ─── Application ──────────────────────────────────────────────────────────────

async fn run_application(mut config: PipelineConfig) -> Result<()> {
    let project_dir = app_dir()?;
    let scripts_dir = project_dir.join("scripts");

    // ── Load and initialize all model bridges (done once) ─────────────────────
    let models = load_models(&config, &scripts_dir).await?;

    // ── Enumerate audio devices for the UI ─────────────────────────────────────
    let output_device_names = list_output_device_names();
    let input_device_names = list_input_device_names();

    // ── Create tray UI ─────────────────────────────────────────────────────────
    let mut tray = TrayUi::new()
        .map_err(|e| anyhow::anyhow!("Tray UI failed: {}", e))?;

    let audio_switch_script = scripts_dir.join("audio_switch.ps1");

    // ── Start both pipelines ───────────────────────────────────────────────────
    let mut pipelines = start_pipelines(&config, &models).await?;
    // Pipelines start paused; user presses Start via tray to activate
    tracing::info!("Pipelines created — use system tray to start.");

    // ── Main event loop ────────────────────────────────────────────────────────
    let mut is_active = false;
    // Saved default device names — restored when the user presses Stop or quits.
    let mut saved_default_output: Option<String> = None;
    let mut saved_default_input: Option<String> = None;

    loop {
        if let Some(action) = tray.process_events() {
            match action {
                TrayAction::Command(cmd) => {
                    match cmd {
                        PipelineCommand::Start => {
                            // Switch system default OUTPUT to VB-Cable so meetings output there.
                            // Loopback captures from CABLE (meeting audio only).
                            match audio_switch::set_default_output_device(
                                &audio_switch_script, VIRTUAL_CABLE_NAME,
                            ) {
                                Ok(previous) => {
                                    tracing::info!(
                                        "Switched default output: \"{}\" → \"{}\"",
                                        previous, VIRTUAL_CABLE_NAME,
                                    );
                                    saved_default_output = Some(previous);
                                    config.loopback_device = Some(VIRTUAL_CABLE_NAME.to_string());
                                }
                                Err(e) => {
                                    tracing::warn!("Could not switch output device: {}. \
                                        Falling back to shared device (may cause echo).", e);
                                }
                            }
                            // Switch system default INPUT to Hi-Fi Cable so meeting
                            // apps read translated mic audio from it.
                            match audio_switch::set_default_input_device(
                                &audio_switch_script, HIFI_CABLE_OUTPUT_NAME,
                            ) {
                                Ok(previous) => {
                                    tracing::info!(
                                        "Switched default input: \"{}\" → \"{}\"",
                                        previous, HIFI_CABLE_OUTPUT_NAME,
                                    );
                                    saved_default_input = Some(previous);
                                }
                                Err(e) => {
                                    tracing::warn!("Could not switch input device: {}. \
                                        Set your meeting app mic to Hi-Fi Cable manually.", e);
                                }
                            }
                            // Recreate pipelines with the new device routing
                            pipelines = restart_pipelines(pipelines, &config, &models, false).await?;
                            is_active = true;
                            tray.set_active(true);
                            pipelines.send_command(PipelineCommand::Start).await;
                        }
                        PipelineCommand::Stop => {
                            is_active = false;
                            tray.set_active(false);
                            pipelines.send_command(PipelineCommand::Stop).await;
                            // Restore original default devices
                            restore_default_output(&audio_switch_script, &mut saved_default_output);
                            restore_default_input(&audio_switch_script, &mut saved_default_input);
                        }
                    }
                }

                TrayAction::OpenSettings => {
                    tray.open_settings(&output_device_names, &input_device_names, &config, is_active);
                }

                TrayAction::SetSpeakerSourceLanguage(lang) => {
                    config.speaker_source_language = lang;
                    config.speaker_target_language = opposite_language(lang);
                    save_config(&config);
                    tracing::info!(
                        "Fone direction: {} → {}",
                        config.speaker_source_language.display_name(),
                        config.speaker_target_language.display_name(),
                    );
                    pipelines = restart_pipelines(pipelines, &config, &models, is_active).await?;
                }

                TrayAction::SetMicSourceLanguage(lang) => {
                    config.mic_source_language = lang;
                    config.mic_target_language = opposite_language(lang);
                    save_config(&config);
                    tracing::info!(
                        "Mic direction: {} → {}",
                        config.mic_source_language.display_name(),
                        config.mic_target_language.display_name(),
                    );
                    pipelines = restart_pipelines(pipelines, &config, &models, is_active).await?;
                }

                TrayAction::SetHeadphonesDevice(name) => {
                    // Headphones = where user hears TTS + passthrough audio.
                    // Loopback device is managed automatically (CABLE when active).
                    config.headphones_device = Some(name);
                    save_config(&config);
                    pipelines = restart_pipelines(pipelines, &config, &models, is_active).await?;
                }

                TrayAction::SetMicDevice(name) => {
                    config.mic_device = Some(name);
                    save_config(&config);
                    pipelines = restart_pipelines(pipelines, &config, &models, is_active).await?;
                }

                TrayAction::Quit => {
                    tracing::info!("Shutting down…");
                    // Restore original defaults before exiting
                    restore_default_output(&audio_switch_script, &mut saved_default_output);
                    restore_default_input(&audio_switch_script, &mut saved_default_input);
                    break;
                }
            }
        }

        // Pump Win32 messages — required for the tray icon context menu to appear
        pump_win32_messages();
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }

    tracing::info!("Meeting Translator stopped");
    Ok(())
}

/// Process pending Windows messages so the tray icon context menu works.
#[cfg(windows)]
fn pump_win32_messages() {
    unsafe {
        let mut msg: MSG = std::mem::zeroed();
        while PeekMessageW(&mut msg, std::ptr::null_mut(), 0, 0, PM_REMOVE) != 0 {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }
}

#[cfg(not(windows))]
fn pump_win32_messages() {}

// ─── Model loading ─────────────────────────────────────────────────────────────

async fn load_models(config: &PipelineConfig, scripts_dir: &std::path::Path) -> Result<LoadedModels> {
    let stt_script = scripts_dir.join("stt_bridge.py");
    let translation_script = scripts_dir.join("translation_bridge.py");
    let tts_script = scripts_dir.join("tts_bridge.py");
    let whisper_model: std::path::PathBuf = config.whisper_model.clone().into();

    // ── STT: two concurrent processes (both auto-detect language) ────────────
    tracing::info!("Initializing STT process A (speaker: {})…", config.speaker_source_language.display_name());
    let mut stt_a = WhisperStt::new(stt_script.clone(), whisper_model.clone(), config.speaker_source_language);
    stt_a.initialize().await?;

    tracing::info!("Initializing STT process B (mic: {})…", config.mic_source_language.display_name());
    let mut stt_b = WhisperStt::new(stt_script, whisper_model, config.mic_source_language);
    stt_b.initialize().await?;

    // ── Translators: always load BOTH directions ────────────────────────────
    tracing::info!("Initializing EN → PT translator…");
    let mut translator_en_pt = OpusMtTranslator::new(
        translation_script.clone(),
        TranslationDirection::new(Language::English, Language::Portuguese),
    );
    translator_en_pt.initialize().await?;

    tracing::info!("Initializing PT → EN translator…");
    let mut translator_pt_en = OpusMtTranslator::new(
        translation_script,
        TranslationDirection::new(Language::Portuguese, Language::English),
    );
    translator_pt_en.initialize().await?;

    // ── TTS: always load BOTH voices ────────────────────────────────────────
    tracing::info!("Initializing Portuguese TTS…");
    let mut tts_portuguese = PiperTts::new(tts_script.clone(), Language::Portuguese);
    tts_portuguese.initialize().await?;

    tracing::info!("Initializing English TTS…");
    let mut tts_english = PiperTts::new(tts_script, Language::English);
    tts_english.initialize().await?;

    Ok(LoadedModels {
        stt_a: Arc::new(stt_a),
        stt_b: Arc::new(stt_b),
        translator_en_pt: Arc::new(translator_en_pt),
        translator_pt_en: Arc::new(translator_pt_en),
        tts_portuguese: Arc::new(tts_portuguese),
        tts_english: Arc::new(tts_english),
    })
}

// ─── Pipeline wiring ──────────────────────────────────────────────────────────

async fn start_pipelines(
    config: &PipelineConfig,
    models: &LoadedModels,
) -> Result<ActivePipelines> {
    // Shared echo buffer: both pipelines write translations and check STT
    // against it. Prevents mic TTS from being re-translated by the speaker pipeline.
    let echo_buffer = pipeline::new_echo_buffer();

    let spk_source_lang = config.speaker_source_language;
    let (spk_trans, spk_tts) = models_for_source(spk_source_lang, models);
    let spk_stt = Arc::clone(&models.stt_a);

    let mic_source_lang = config.mic_source_language;
    let (mic_trans, mic_tts) = models_for_source(mic_source_lang, models);
    let mic_stt = Arc::clone(&models.stt_b);

    // ── Speaker pipeline ───────────────────────────────────────────────────────
    let (spk_audio_tx, spk_audio_rx) = mpsc::unbounded_channel::<shared::AudioChunk>();
    let (spk_out_tx, spk_out_rx) = mpsc::unbounded_channel::<shared::AudioChunk>();
    let (spk_cmd_tx, spk_cmd_rx) = mpsc::channel::<PipelineCommand>(8);
    let (spk_metrics_tx, mut spk_metrics_rx) = mpsc::channel::<shared::PipelineMetrics>(64);

    let loopback_device = resolve_output_device(config.loopback_device.as_deref(), "loopback")?;
    let loopback_name = cpal::traits::DeviceTrait::name(&loopback_device)
        .unwrap_or_else(|_| "Unknown".to_string());
    tracing::info!("Speaker loopback from: {}", loopback_name);

    let headphones_device = resolve_output_device(config.headphones_device.as_deref(), "headphones")?;
    let headphones_name = cpal::traits::DeviceTrait::name(&headphones_device)
        .unwrap_or_else(|_| "Unknown".to_string());
    tracing::info!("Speaker TTS output to: {}", headphones_name);

    // When loopback is on VB-Cable (different from headphones), split the
    // loopback into two streams: the existing 16 kHz denoised path for STT
    // and a pristine native-rate pre-denoise path for the mixer passthrough.
    // Otherwise (shared-device setup), no passthrough — user hears the meeting
    // directly on the same device.
    let loopback_is_cable = config.loopback_device.as_deref()
        .map(|d| d.to_lowercase().contains("cable"))
        .unwrap_or(false);

    let (passthrough_tx, passthrough_rx) = mpsc::unbounded_channel::<shared::AudioChunk>();

    let speaker_capture = if loopback_is_cable {
        let loopback = LoopbackCapture::new(loopback_device, config.chunk_duration_ms);
        let capture = loopback
            .start_with_raw(spk_audio_tx, passthrough_tx)
            .map_err(|e| anyhow::anyhow!("Loopback capture failed: {}", e))?;
        tracing::info!("Passthrough active: raw meeting audio → mixer");
        capture
    } else {
        // No passthrough — drop the tx so the mixer's passthrough buffer stays empty.
        drop(passthrough_tx);
        let loopback = LoopbackCapture::new(loopback_device, config.chunk_duration_ms);
        loopback
            .start(spk_audio_tx)
            .map_err(|e| anyhow::anyhow!("Loopback capture failed: {}", e))?
    };

    let mixer = MixerPlayback::new(headphones_device);
    let speaker_mixer = mixer
        .start(passthrough_rx, spk_out_rx)
        .map_err(|e| anyhow::anyhow!("Mixer playback failed: {}", e))?;

    let speaker_pipeline = SpeakerPipeline::new(
        "Speaker", spk_stt, spk_trans, spk_tts, spk_source_lang, 2.5, echo_buffer.clone(),
    );
    tokio::spawn(async move {
        speaker_pipeline.run(spk_audio_rx, spk_out_tx, spk_cmd_rx, spk_metrics_tx).await;
    });
    tokio::spawn(async move {
        while let Some(metric) = spk_metrics_rx.recv().await {
            if metric.stage_name == "total" {
                tracing::info!("[Speaker] latency: {}ms", metric.processing_duration.as_millis());
            }
        }
    });

    // ── Mic pipeline ───────────────────────────────────────────────────────────
    let (mic_audio_tx, mic_audio_rx) = mpsc::unbounded_channel::<shared::AudioChunk>();
    let (mic_out_tx, mic_out_rx) = mpsc::unbounded_channel::<shared::AudioChunk>();
    let (mic_cmd_tx, mic_cmd_rx) = mpsc::channel::<PipelineCommand>(8);
    let (mic_metrics_tx, mut mic_metrics_rx) = mpsc::channel::<shared::PipelineMetrics>(64);

    let mic_device = resolve_input_device(config.mic_device.as_deref())?;
    let mic_device_name = cpal::traits::DeviceTrait::name(&mic_device)
        .unwrap_or_else(|_| "Unknown".to_string());
    tracing::info!("Mic capture from: {}", mic_device_name);

    let mic_capture_node = AudioCapture::new(mic_device, config.chunk_duration_ms);
    let mic_capture = mic_capture_node
        .start(mic_audio_tx)
        .map_err(|e| anyhow::anyhow!("Mic capture failed: {}", e))?;

    let virtual_mic_device =
        resolve_output_device(Some(config.effective_virtual_mic()), "virtual mic")?;
    let virtual_mic_name = cpal::traits::DeviceTrait::name(&virtual_mic_device)
        .unwrap_or_else(|_| "Unknown".to_string());
    tracing::info!("Mic TTS output to: {}", virtual_mic_name);

    let mic_playback_node = AudioPlayback::new(virtual_mic_device);
    let mic_playback = mic_playback_node
        .start(mic_out_rx)
        .map_err(|e| anyhow::anyhow!("Mic playback failed: {}", e))?;

    // Mic pipeline: 2.5s flush — match speaker pipeline
    let mic_pipeline = SpeakerPipeline::new(
        "Mic", mic_stt, mic_trans, mic_tts, mic_source_lang, 2.5, echo_buffer,
    );
    tokio::spawn(async move {
        mic_pipeline.run(mic_audio_rx, mic_out_tx, mic_cmd_rx, mic_metrics_tx).await;
    });
    tokio::spawn(async move {
        while let Some(metric) = mic_metrics_rx.recv().await {
            if metric.stage_name == "total" {
                tracing::info!("[Mic] latency: {}ms", metric.processing_duration.as_millis());
            }
        }
    });

    Ok(ActivePipelines {
        _speaker_capture: speaker_capture,
        _speaker_mixer: speaker_mixer,
        _mic_capture: mic_capture,
        _mic_playback: mic_playback,
        speaker_cmd_tx: spk_cmd_tx,
        mic_cmd_tx: mic_cmd_tx,
    })
}

/// Drop old pipelines (streams stop, tasks exit) and start fresh ones.
async fn restart_pipelines(
    old: ActivePipelines,
    config: &PipelineConfig,
    models: &LoadedModels,
    was_active: bool,
) -> Result<ActivePipelines> {
    drop(old);
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    let pipelines = start_pipelines(config, models).await?;

    if was_active {
        pipelines.send_command(PipelineCommand::Start).await;
    }

    Ok(pipelines)
}

// ─── Pipeline model helpers ───────────────────────────────────────────────────

/// Pick the translator and TTS models for a pipeline based on its source language.
/// - English source → uses the EN→PT translator and Portuguese TTS voice.
/// - Portuguese source → uses the PT→EN translator and English TTS voice.
fn models_for_source(
    source: Language,
    models: &LoadedModels,
) -> (Arc<OpusMtTranslator>, Arc<PiperTts>) {
    match source {
        Language::English => (
            Arc::clone(&models.translator_en_pt),
            Arc::clone(&models.tts_portuguese),
        ),
        Language::Portuguese => (
            Arc::clone(&models.translator_pt_en),
            Arc::clone(&models.tts_english),
        ),
    }
}

fn opposite_language(lang: Language) -> Language {
    match lang {
        Language::English => Language::Portuguese,
        Language::Portuguese => Language::English,
    }
}

// ─── Device resolution helpers ────────────────────────────────────────────────

fn resolve_output_device(name: Option<&str>, role: &str) -> Result<cpal::Device> {
    match name {
        Some(n) => device::find_output_device_by_name(n)
            .map_err(|e| anyhow::anyhow!("Output device '{}' ({}) not found: {}", n, role, e)),
        None => device::get_default_output_device()
            .map_err(|e| anyhow::anyhow!("No default output device for {}: {}", role, e)),
    }
}

fn resolve_input_device(name: Option<&str>) -> Result<cpal::Device> {
    match name {
        Some(n) => device::find_input_device_by_name(n)
            .map_err(|e| anyhow::anyhow!("Input device '{}' not found: {}", n, e)),
        None => device::get_default_input_device()
            .map_err(|e| anyhow::anyhow!("No default input device: {}", e)),
    }
}

fn list_output_device_names() -> Vec<String> {
    device::list_output_devices()
        .unwrap_or_default()
        .into_iter()
        .map(|d| d.name)
        .collect()
}

fn list_input_device_names() -> Vec<String> {
    device::list_input_devices()
        .unwrap_or_default()
        .into_iter()
        .map(|d| d.name)
        .collect()
}

// ─── Audio device switching ──────────────────────────────────────────────────

/// Restore the Windows default output device to what the user had before.
fn restore_default_output(script_path: &std::path::Path, saved: &mut Option<String>) {
    if let Some(device_name) = saved.take() {
        match audio_switch::set_default_output_device(script_path, &device_name) {
            Ok(_) => tracing::info!("Restored default output: \"{}\"", device_name),
            Err(e) => tracing::warn!("Failed to restore default output: {}", e),
        }
    }
}

/// Restore the Windows default input (microphone) device to what the user had before.
fn restore_default_input(script_path: &std::path::Path, saved: &mut Option<String>) {
    if let Some(device_name) = saved.take() {
        match audio_switch::set_default_input_device(script_path, &device_name) {
            Ok(_) => tracing::info!("Restored default input: \"{}\"", device_name),
            Err(e) => tracing::warn!("Failed to restore default input: {}", e),
        }
    }
}

// ─── Path resolution ─────────────────────────────────────────────────────────

/// Returns the application's base directory.
///
/// - **Installed mode** (exe lives outside a `target/` build dir):
///   uses the exe's own directory (e.g. `C:\Program Files\Meeting Translator\`).
/// - **Dev mode** (`cargo run` — exe lives inside `target\release\` or `target\debug\`):
///   uses the current working directory (the project root).
fn app_dir() -> Result<std::path::PathBuf> {
    let exe_path = std::env::current_exe()?;
    let exe_dir = exe_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("Could not determine executable directory"))?;

    // Detect dev mode: exe is inside a `target\{profile}\` directory
    let is_dev = exe_dir
        .components()
        .any(|c| c.as_os_str() == "target");

    if is_dev {
        Ok(std::env::current_dir()?)
    } else {
        Ok(exe_dir.to_path_buf())
    }
}

// ─── Config I/O ───────────────────────────────────────────────────────────────

fn load_config() -> Result<PipelineConfig> {
    let config_path = app_dir()?.join("config.toml");
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

fn save_config(config: &PipelineConfig) {
    let config_path = match app_dir() {
        Ok(dir) => dir.join("config.toml"),
        Err(e) => {
            tracing::warn!("Could not determine app dir for config save: {}", e);
            return;
        }
    };
    match toml::to_string_pretty(config) {
        Ok(content) => {
            if let Err(e) = std::fs::write(&config_path, content) {
                tracing::warn!("Failed to save config: {}", e);
            } else {
                tracing::info!("Config saved to {}", config_path.display());
            }
        }
        Err(e) => tracing::warn!("Failed to serialize config: {}", e),
    }
}
