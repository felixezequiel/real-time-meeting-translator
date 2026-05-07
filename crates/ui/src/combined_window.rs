//! Multi-viewport host (ADR 0013, Phase 3.1).
//!
//! Single eframe::App that owns BOTH:
//!   - the subtitle overlay (always-on-top, transparent) as the
//!     primary viewport;
//!   - the settings panel as a deferred secondary viewport, shown on
//!     demand when the user clicks "Configurações" in the tray menu.
//!
//! Solves the field-confirmed limitation that two `eframe::run_native`
//! event loops cannot coexist on Windows (winit 0.30): with this
//! design only ONE event loop runs, hosting both viewports under a
//! shared `egui::Context`.
//!
//! When `overlay_enabled = false`, the primary viewport is created
//! invisible (1×1 px, hidden) so the user only sees the settings
//! window when they ask for it. This keeps the multi-viewport path
//! viable even when the user does not want subtitles.

use eframe::egui;
use std::sync::mpsc as std_mpsc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use shared::StageMetricsAggregator;

use crate::settings_window::{configure_style, SettingsApp, SettingsInit};
use crate::subtitle_overlay::{SubtitleMessage, SubtitleState};
use crate::TrayAction;

const OVERLAY_WIDTH: f32 = 1100.0;
const OVERLAY_HEIGHT: f32 = 200.0;
const OVERLAY_INITIAL_X: f32 = 410.0;
const OVERLAY_INITIAL_Y: f32 = 760.0;

const SETTINGS_WIDTH: f32 = 460.0;
const SETTINGS_HEIGHT: f32 = 540.0;
const SETTINGS_MIN_WIDTH: f32 = 420.0;
const SETTINGS_MIN_HEIGHT: f32 = 500.0;

/// Repaint cadence for the overlay (≈30 Hz).
const OVERLAY_REPAINT_INTERVAL: Duration = Duration::from_millis(33);

/// Idle repaint cadence when the overlay is disabled. Without this,
/// `update()` only runs on input events, and the `show_settings_rx`
/// channel never gets drained → clicking Configurações appears to do
/// nothing. 4 Hz is enough to feel instant on the click and burns
/// negligible CPU.
const IDLE_POLL_INTERVAL: Duration = Duration::from_millis(250);

pub struct CombinedApp {
    /// SettingsApp is held directly (no Arc<Mutex<>>) because its
    /// embedded `VoiceRecorder` wraps WASAPI handles that are !Send,
    /// which forced us to use `show_viewport_immediate` (FnMut, sync,
    /// no Send/Sync bound) instead of `show_viewport_deferred`.
    settings: SettingsApp,
    subtitle: SubtitleState,
    show_settings_rx: std_mpsc::Receiver<SettingsInit>,
    settings_visible: bool,
    subtitle_msg_rx: std_mpsc::Receiver<SubtitleMessage>,
    overlay_enabled: bool,
}

impl eframe::App for CombinedApp {
    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        // Primary viewport is the overlay → transparent. Even when
        // overlay_enabled = false the viewport is hidden, so the
        // clear color is irrelevant; transparent is safe in both cases.
        [0.0, 0.0, 0.0, 0.0]
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Drain "show settings" signals from the tray and update the
        // settings app's state for the upcoming render.
        while let Ok(init) = self.show_settings_rx.try_recv() {
            self.settings.apply_init(init);
            self.settings_visible = true;
        }

        // Drain subtitle messages into the overlay state.
        while let Ok(msg) = self.subtitle_msg_rx.try_recv() {
            self.subtitle.push(msg);
        }

        // Primary viewport — overlay (or invisible dummy).
        if self.overlay_enabled {
            self.subtitle.prune_invisible();
            self.subtitle.render(ctx);
            ctx.request_repaint_after(OVERLAY_REPAINT_INTERVAL);
        } else {
            // Without an active overlay we still need to wake up
            // periodically to drain show_settings_rx and respond to
            // the user clicking Configurações.
            ctx.request_repaint_after(IDLE_POLL_INTERVAL);
        }

        // Secondary viewport — settings, shown on demand.
        // `show_viewport_immediate` runs SYNCHRONOUSLY inside this
        // update call, so the closure can borrow `&mut self.settings`
        // without violating Send/Sync (which `show_viewport_deferred`
        // would require — and `SettingsApp` contains a !Send
        // VoiceRecorder, ruling deferred out).
        if self.settings_visible {
            let mut close_requested = false;
            ctx.show_viewport_immediate(
                egui::ViewportId::from_hash_of("meeting-translator-settings"),
                egui::ViewportBuilder::default()
                    .with_title("Configurações")
                    .with_inner_size([SETTINGS_WIDTH, SETTINGS_HEIGHT])
                    .with_min_inner_size([SETTINGS_MIN_WIDTH, SETTINGS_MIN_HEIGHT])
                    .with_resizable(true)
                    .with_always_on_top(),
                |ctx, _class| {
                    self.settings.render_ui(ctx);
                    if ctx.input(|i| i.viewport().close_requested()) {
                        close_requested = true;
                    }
                },
            );
            if close_requested {
                self.settings_visible = false;
            }
        }
    }
}

/// Spawn the multi-viewport host on its own OS thread. Returns a
/// `(JoinHandle, settings_show_tx, subtitle_tx)` triple:
///   - `JoinHandle` joins when the app quits.
///   - `settings_show_tx`: tray writes a fresh `SettingsInit` here to
///     pop up the settings viewport.
///   - `subtitle_tx`: pipeline pushes translated phrases here for the
///     overlay.
pub fn spawn_combined(
    settings_init: SettingsInit,
    action_tx: std_mpsc::Sender<TrayAction>,
    metrics: Arc<StageMetricsAggregator>,
    overlay_enabled: bool,
) -> Result<(
    std::thread::JoinHandle<()>,
    std_mpsc::Sender<SettingsInit>,
    std_mpsc::Sender<SubtitleMessage>,
), String> {
    let (show_settings_tx, show_settings_rx) = std_mpsc::channel::<SettingsInit>();
    let (subtitle_tx, subtitle_rx) = std_mpsc::channel::<SubtitleMessage>();

    let handle = std::thread::Builder::new()
        .name("combined-ui".to_string())
        .spawn(move || {
            // SettingsApp expects its own pending_show Mutex even
            // though the combined path bypasses the watcher thread.
            // Pass an unused Mutex to keep its constructor happy.
            let pending_show = Arc::new(Mutex::new(None));
            let settings_app = SettingsApp::new(
                settings_init,
                action_tx,
                pending_show,
                metrics,
            );

            let app = CombinedApp {
                settings: settings_app,
                subtitle: SubtitleState::new(),
                show_settings_rx,
                settings_visible: false,
                subtitle_msg_rx: subtitle_rx,
                overlay_enabled,
            };

            // Primary viewport = overlay (transparent always-on-top).
            // When overlay_enabled = false, the window is created
            // invisible and tiny so the user never notices it; only
            // the secondary settings viewport will show.
            let viewport = if overlay_enabled {
                egui::ViewportBuilder::default()
                    .with_decorations(false)
                    .with_transparent(true)
                    .with_always_on_top()
                    .with_resizable(false)
                    .with_inner_size([OVERLAY_WIDTH, OVERLAY_HEIGHT])
                    .with_position([OVERLAY_INITIAL_X, OVERLAY_INITIAL_Y])
                    .with_title("Legenda da tradução")
            } else {
                egui::ViewportBuilder::default()
                    .with_decorations(false)
                    .with_transparent(true)
                    .with_resizable(false)
                    .with_inner_size([1.0, 1.0])
                    .with_visible(false)
                    .with_title("meeting-translator")
            };

            let options = eframe::NativeOptions {
                viewport,
                centered: false,
                event_loop_builder: Some(Box::new(|builder| {
                    #[cfg(target_os = "windows")]
                    {
                        use winit::platform::windows::EventLoopBuilderExtWindows;
                        builder.with_any_thread(true);
                    }
                    let _ = builder;
                })),
                ..Default::default()
            };

            if let Err(e) = eframe::run_native(
                "meeting-translator",
                options,
                Box::new(|cc| {
                    configure_style(&cc.egui_ctx);
                    Ok(Box::new(app))
                }),
            ) {
                tracing::error!("Combined UI event loop ended: {}", e);
            }
        })
        .map_err(|e| format!("Failed to spawn combined-ui thread: {}", e))?;

    Ok((handle, show_settings_tx, subtitle_tx))
}
