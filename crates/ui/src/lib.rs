pub mod settings_window;

use shared::{Language, PipelineCommand, PipelineConfig};
use tray_icon::menu::{Menu, MenuEvent, MenuId, MenuItem, PredefinedMenuItem};
use tray_icon::{Icon, TrayIcon, TrayIconBuilder};
use tracing;

use std::sync::mpsc as std_mpsc;

const ICON_SIZE: u32 = 32;

// ─── Public types ─────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum TrayAction {
    Command(PipelineCommand),
    /// Change the source language of the Fone (speaker/loopback) pipeline independently.
    SetSpeakerSourceLanguage(Language),
    /// Change the source language of the Mic pipeline independently.
    SetMicSourceLanguage(Language),
    SetHeadphonesDevice(String),
    SetMicDevice(String),
    OpenSettings,
    Quit,
}

// ─── Tray UI ──────────────────────────────────────────────────────────────────

pub struct TrayUi {
    tray_icon: TrayIcon,
    settings_id: MenuId,
    quit_id: MenuId,
    /// Receives actions from the settings egui window (runs in a separate thread)
    window_action_rx: std_mpsc::Receiver<TrayAction>,
    /// Sender cloned into each settings window
    window_action_tx: std_mpsc::Sender<TrayAction>,
    /// Handle to the settings window thread (if open)
    settings_thread: Option<std::thread::JoinHandle<()>>,
}

impl TrayUi {
    pub fn new() -> Result<Self, String> {
        let menu = Menu::new();

        let settings_item = MenuItem::new("⚙  Configurações", true, None);
        let settings_id = settings_item.id().clone();

        let quit_item = MenuItem::new("✕  Sair", true, None);
        let quit_id = quit_item.id().clone();

        let _ = menu.append(&settings_item);
        let _ = menu.append(&PredefinedMenuItem::separator());
        let _ = menu.append(&quit_item);

        let tray_icon = TrayIconBuilder::new()
            .with_menu(Box::new(menu))
            .with_tooltip("Meeting Translator")
            .with_icon(create_icon(false))
            .build()
            .map_err(|e| format!("Failed to create tray icon: {}", e))?;

        let (window_action_tx, window_action_rx) = std_mpsc::channel();

        tracing::info!("System tray UI created");

        Ok(Self {
            tray_icon,
            settings_id,
            quit_id,
            window_action_rx,
            window_action_tx,
            settings_thread: None,
        })
    }

    /// Poll for events from both the tray menu and the settings window.
    pub fn process_events(&mut self) -> Option<TrayAction> {
        // Check actions coming from the egui settings window
        if let Ok(action) = self.window_action_rx.try_recv() {
            return Some(action);
        }

        // Check tray menu clicks
        let event = MenuEvent::receiver().try_recv().ok()?;
        let id = event.id();

        if id == &self.settings_id {
            return Some(TrayAction::OpenSettings);
        }

        if id == &self.quit_id {
            return Some(TrayAction::Quit);
        }

        None
    }

    /// Open the egui settings window (if not already open).
    pub fn open_settings(
        &mut self,
        output_devices: &[String],
        input_devices: &[String],
        config: &PipelineConfig,
        is_active: bool,
    ) {
        // Check if the window thread is still running
        if let Some(handle) = &self.settings_thread {
            if !handle.is_finished() {
                tracing::info!("Settings window already open");
                return;
            }
        }

        let init = settings_window::SettingsInit {
            output_devices: output_devices.to_vec(),
            input_devices: input_devices.to_vec(),
            selected_mic: config.mic_device.clone().unwrap_or_default(),
            selected_headphones: config.headphones_device.clone().unwrap_or_default(),
            mic_source_lang: config.mic_source_language,
            speaker_source_lang: config.speaker_source_language,
            is_active,
        };

        let handle = settings_window::open(init, self.window_action_tx.clone());
        self.settings_thread = Some(handle);
        tracing::info!("Settings window opened");
    }

    pub fn set_active(&self, active: bool) {
        let _ = self.tray_icon.set_icon(Some(create_icon(active)));
        let tooltip = if active {
            "Meeting Translator — Ativo"
        } else {
            "Meeting Translator — Inativo"
        };
        let _ = self.tray_icon.set_tooltip(Some(tooltip));
    }
}

// ─── Icon helpers ─────────────────────────────────────────────────────────────

fn create_icon(active: bool) -> Icon {
    let mut rgba = Vec::with_capacity((ICON_SIZE * ICON_SIZE * 4) as usize);
    let (r, g, b) = if active { (40, 180, 80) } else { (60, 120, 200) };
    for _ in 0..ICON_SIZE * ICON_SIZE {
        rgba.push(r);
        rgba.push(g);
        rgba.push(b);
        rgba.push(255);
    }
    Icon::from_rgba(rgba, ICON_SIZE, ICON_SIZE).expect("Failed to create icon")
}
