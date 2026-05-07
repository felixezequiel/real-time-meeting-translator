//! Always-on-top subtitle overlay (ADR 0013, Phase 2).
//!
//! A frameless transparent eframe window pinned to the bottom of the
//! primary screen. The pipeline pushes translated phrases here so the
//! user can read while the TTS audio catches up, addressing the
//! "delay para fazer sentido na frase" requirement without forcing a
//! 3-5 s audio gap.
//!
//! The overlay runs in its own thread (eframe takes the main thread),
//! and consumes `SubtitleMessage` via a `std::sync::mpsc` channel so
//! the pipeline doesn't depend on tokio runtime context.

use eframe::egui::{self, Color32, FontId, RichText, Stroke};
use std::collections::VecDeque;
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// One subtitle event delivered to the overlay.
#[derive(Debug, Clone)]
pub struct SubtitleMessage {
    pub source_text: String,
    pub translated_text: String,
    pub speaker: String,
}

/// How long a phrase stays fully visible before fading out.
const VISIBLE_DURATION: Duration = Duration::from_secs(8);
/// Linear fade duration after `VISIBLE_DURATION` elapses.
const FADE_DURATION: Duration = Duration::from_millis(800);
/// Maximum number of phrases shown stacked at once.
const MAX_VISIBLE_LINES: usize = 3;
/// Width of the overlay window (px).
const WINDOW_WIDTH: f32 = 1100.0;
/// Height of the overlay window (px).
const WINDOW_HEIGHT: f32 = 200.0;
/// Initial Y position from the top (the user can drag the window
/// afterwards; this just lands it in a sensible default location for a
/// typical 1080p display — bottom third).
const INITIAL_Y_FROM_TOP: f32 = 760.0;
/// Initial X position. Centred for a 1920-wide display.
const INITIAL_X: f32 = 410.0;

#[derive(Clone)]
struct DisplayedLine {
    speaker: String,
    text: String,
    arrived_at: Instant,
}

impl DisplayedLine {
    fn alpha(&self) -> f32 {
        let age = self.arrived_at.elapsed();
        if age < VISIBLE_DURATION {
            1.0
        } else {
            let fade_age = age - VISIBLE_DURATION;
            if fade_age >= FADE_DURATION {
                0.0
            } else {
                1.0 - fade_age.as_secs_f32() / FADE_DURATION.as_secs_f32()
            }
        }
    }
}

/// Pure rendering state for the subtitle overlay. Extracted so the
/// combined multi-viewport host (`combined_window.rs`) can render it
/// in a primary viewport without owning an eframe::App per overlay.
pub struct SubtitleState {
    lines: VecDeque<DisplayedLine>,
}

impl Default for SubtitleState {
    fn default() -> Self {
        Self {
            lines: VecDeque::new(),
        }
    }
}

impl SubtitleState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, msg: SubtitleMessage) {
        self.lines.push_back(DisplayedLine {
            speaker: msg.speaker,
            text: msg.translated_text,
            arrived_at: Instant::now(),
        });
        while self.lines.len() > MAX_VISIBLE_LINES {
            self.lines.pop_front();
        }
    }

    pub fn prune_invisible(&mut self) {
        while let Some(line) = self.lines.front() {
            if line.alpha() <= 0.0 {
                self.lines.pop_front();
            } else {
                break;
            }
        }
    }

    pub fn render(&self, ctx: &egui::Context) {
        egui::CentralPanel::default()
            .frame(egui::Frame::none())
            .show(ctx, |ui| {
                ui.set_min_height(WINDOW_HEIGHT);
                ui.with_layout(egui::Layout::bottom_up(egui::Align::Center), |ui| {
                    for line in self.lines.iter().rev() {
                        let alpha = line.alpha();
                        if alpha <= 0.0 {
                            continue;
                        }
                        let alpha_byte = (alpha * 255.0) as u8;
                        let text_color = Color32::from_rgba_unmultiplied(
                            255, 255, 255, alpha_byte,
                        );
                        let bg_color = Color32::from_rgba_unmultiplied(
                            0, 0, 0,
                            (alpha * 200.0) as u8,
                        );
                        let label = RichText::new(format!("{}: {}", line.speaker, line.text))
                            .font(FontId::proportional(28.0))
                            .color(text_color);
                        egui::Frame::none()
                            .fill(bg_color)
                            .stroke(Stroke::NONE)
                            .inner_margin(egui::Margin::symmetric(16.0, 8.0))
                            .rounding(egui::Rounding::same(6.0))
                            .show(ui, |ui| {
                                ui.label(label);
                            });
                        ui.add_space(4.0);
                    }
                });
            });
    }
}

/// Standalone overlay app — kept for the legacy `spawn_overlay` path.
/// The combined-window path uses `SubtitleState` directly.
struct SubtitleApp {
    rx: mpsc::Receiver<SubtitleMessage>,
    state: SubtitleState,
}

impl SubtitleApp {
    fn new(rx: mpsc::Receiver<SubtitleMessage>) -> Self {
        Self {
            rx,
            state: SubtitleState::new(),
        }
    }
}

impl eframe::App for SubtitleApp {
    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        // Fully transparent — the per-frame painter draws its own
        // semi-opaque background only behind text.
        [0.0, 0.0, 0.0, 0.0]
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        while let Ok(msg) = self.rx.try_recv() {
            self.state.push(msg);
        }
        self.state.prune_invisible();
        ctx.request_repaint_after(Duration::from_millis(33));
        self.state.render(ctx);
    }
}

/// Spawn the overlay window on its own OS thread. The returned sender
/// is the channel the pipeline pushes phrases onto. The thread joins
/// when the user closes the window or the app exits.
pub fn spawn_overlay() -> Result<mpsc::Sender<SubtitleMessage>, String> {
    let (tx, rx) = mpsc::channel::<SubtitleMessage>();

    std::thread::Builder::new()
        .name("subtitle-overlay".to_string())
        .spawn(move || {
            let viewport = egui::ViewportBuilder::default()
                .with_decorations(false)
                .with_transparent(true)
                .with_always_on_top()
                .with_resizable(false)
                .with_inner_size([WINDOW_WIDTH, WINDOW_HEIGHT])
                .with_position([INITIAL_X, INITIAL_Y_FROM_TOP])
                .with_title("Legenda da tradução");

            let options = eframe::NativeOptions {
                viewport,
                centered: false,
                // Same pattern as settings_window: winit on Windows
                // refuses to start an event loop off the main thread
                // by default. The settings already runs on its own
                // thread via this opt-in; the overlay needs the same
                // override so it can coexist on a second worker.
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

            let app = SubtitleApp::new(rx);
            if let Err(e) = eframe::run_native(
                "subtitle-overlay",
                options,
                Box::new(|_cc| Ok(Box::new(app))),
            ) {
                tracing::error!("Subtitle overlay event loop ended: {}", e);
            }
        })
        .map_err(|e| format!("Failed to spawn subtitle overlay thread: {}", e))?;

    Ok(tx)
}
