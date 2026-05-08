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
    /// Same id across every event of one streamed phrase. The overlay
    /// updates the existing line in place when it sees a repeat,
    /// instead of stacking each fragment on top of the previous one.
    pub phrase_id: u64,
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
    phrase_id: u64,
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

    /// Ingest one subtitle event. If the most recent line already
    /// belongs to the same `phrase_id` (i.e. this is another fragment
    /// of the streaming translation already on screen), update its
    /// text and refresh the timestamp instead of stacking a new line.
    /// New `phrase_id` becomes a fresh line at the bottom.
    pub fn push(&mut self, msg: SubtitleMessage) {
        if let Some(last) = self.lines.back_mut() {
            if last.phrase_id == msg.phrase_id {
                last.text = msg.translated_text;
                last.speaker = msg.speaker;
                last.arrived_at = Instant::now();
                return;
            }
        }
        self.lines.push_back(DisplayedLine {
            phrase_id: msg.phrase_id,
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
                // Whole-panel drag handle: clicking-and-dragging
                // anywhere on the overlay (including the text and
                // empty space) repositions the window via winit's
                // StartDrag. This makes the frameless transparent
                // window feel like a free-floating widget the user
                // can move out of their content area at will.
                let drag_response = ui.interact(
                    ui.max_rect(),
                    egui::Id::new("subtitle-overlay-drag"),
                    egui::Sense::click_and_drag(),
                );
                if drag_response.dragged() {
                    ctx.send_viewport_cmd(egui::ViewportCommand::StartDrag);
                }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(phrase_id: u64, text: &str) -> SubtitleMessage {
        SubtitleMessage {
            phrase_id,
            source_text: String::new(),
            translated_text: text.to_string(),
            speaker: "Speaker".to_string(),
        }
    }

    #[test]
    fn push_replaces_line_when_phrase_id_matches() {
        let mut state = SubtitleState::new();
        state.push(msg(1, "Bem-vindo"));
        state.push(msg(1, "Bem-vindo ao primeiro"));
        state.push(msg(1, "Bem-vindo ao primeiro episódio"));
        // Three streaming fragments of the SAME phrase must collapse
        // to ONE line on screen — not stack three growing copies on
        // top of each other.
        assert_eq!(state.lines.len(), 1);
        assert_eq!(state.lines[0].text, "Bem-vindo ao primeiro episódio");
    }

    #[test]
    fn push_starts_new_line_when_phrase_id_changes() {
        let mut state = SubtitleState::new();
        state.push(msg(1, "Primeira frase."));
        state.push(msg(2, "Segunda frase."));
        assert_eq!(state.lines.len(), 2);
        assert_eq!(state.lines[0].text, "Primeira frase.");
        assert_eq!(state.lines[1].text, "Segunda frase.");
    }

    #[test]
    fn push_caps_total_lines_to_max_visible() {
        let mut state = SubtitleState::new();
        for id in 1..=10 {
            state.push(msg(id, &format!("Phrase {}", id)));
        }
        assert_eq!(state.lines.len(), MAX_VISIBLE_LINES);
        // Oldest evict first — most recent IDs survive.
        assert_eq!(state.lines.back().unwrap().text, "Phrase 10");
    }

    #[test]
    fn push_refreshes_arrived_at_on_update() {
        // The replace path must reset the timestamp so the line
        // doesn't fade out mid-stream while fresh fragments are
        // still being delivered.
        let mut state = SubtitleState::new();
        state.push(msg(7, "first chunk"));
        let initial_ts = state.lines[0].arrived_at;
        std::thread::sleep(std::time::Duration::from_millis(20));
        state.push(msg(7, "first chunk + more"));
        let updated_ts = state.lines[0].arrived_at;
        assert!(updated_ts > initial_ts);
    }
}
