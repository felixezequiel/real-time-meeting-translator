use eframe::egui;
use shared::{Language, PipelineCommand};
use std::sync::mpsc as std_mpsc;

use crate::TrayAction;

// ─── Discord-inspired color palette ───────────────────────────────────────────

const BG_PRIMARY: egui::Color32 = egui::Color32::from_rgb(32, 34, 37);
const BG_SECONDARY: egui::Color32 = egui::Color32::from_rgb(47, 49, 54);
const BG_CARD: egui::Color32 = egui::Color32::from_rgb(54, 57, 63);
const BG_INPUT: egui::Color32 = egui::Color32::from_rgb(24, 25, 28);
const TEXT_PRIMARY: egui::Color32 = egui::Color32::from_rgb(220, 221, 222);
const TEXT_MUTED: egui::Color32 = egui::Color32::from_rgb(114, 118, 125);
const TEXT_HEADER: egui::Color32 = egui::Color32::from_rgb(185, 187, 190);
const ACCENT_GREEN: egui::Color32 = egui::Color32::from_rgb(67, 181, 129);
const ACCENT_RED: egui::Color32 = egui::Color32::from_rgb(240, 71, 71);
const ACCENT_BLUE: egui::Color32 = egui::Color32::from_rgb(88, 101, 242);
const ACCENT_ARROW: egui::Color32 = egui::Color32::from_rgb(114, 118, 125);

// ─── Init data ────────────────────────────────────────────────────────────────

pub struct SettingsInit {
    pub output_devices: Vec<String>,
    pub input_devices: Vec<String>,
    pub selected_mic: String,
    pub selected_headphones: String,
    /// Language the user speaks into the mic (mic pipeline source)
    pub mic_source_lang: Language,
    /// Language the incoming audio is in, e.g. EN for YouTube (speaker pipeline source)
    pub speaker_source_lang: Language,
    pub is_active: bool,
}

// ─── Window entry point ───────────────────────────────────────────────────────

pub fn open(
    init: SettingsInit,
    action_tx: std_mpsc::Sender<TrayAction>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let options = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([480.0, 360.0])
                .with_min_inner_size([400.0, 300.0])
                .with_resizable(true)
                .with_always_on_top()
                .with_title_shown(false)
                .with_decorations(false),
            event_loop_builder: Some(Box::new(|builder| {
                use winit::platform::windows::EventLoopBuilderExtWindows;
                builder.with_any_thread(true);
            })),
            ..Default::default()
        };

        let tx = action_tx;
        let _ = eframe::run_native(
            "Meeting Translator",
            options,
            Box::new(move |cc| {
                configure_style(&cc.egui_ctx);
                Ok(Box::new(SettingsApp::new(init, tx)))
            }),
        );
    })
}

// ─── App state ────────────────────────────────────────────────────────────────

struct SettingsApp {
    output_devices: Vec<String>,
    input_devices: Vec<String>,
    mic_idx: usize,
    headphones_idx: usize,
    mic_source_lang: Language,
    speaker_source_lang: Language,
    is_active: bool,
    action_tx: std_mpsc::Sender<TrayAction>,
    drag_origin: Option<egui::Pos2>,
    window_pos: Option<egui::Pos2>,
}

impl SettingsApp {
    fn new(init: SettingsInit, action_tx: std_mpsc::Sender<TrayAction>) -> Self {
        let mic_idx = init
            .input_devices
            .iter()
            .position(|d| *d == init.selected_mic)
            .unwrap_or(0);
        let headphones_idx = init
            .output_devices
            .iter()
            .position(|d| *d == init.selected_headphones)
            .unwrap_or(0);

        Self {
            output_devices: init.output_devices,
            input_devices: init.input_devices,
            mic_idx,
            headphones_idx,
            mic_source_lang: init.mic_source_lang,
            speaker_source_lang: init.speaker_source_lang,
            is_active: init.is_active,
            action_tx,
            drag_origin: None,
            window_pos: None,
        }
    }

    fn send(&self, action: TrayAction) {
        let _ = self.action_tx.send(action);
    }

    fn mic_name(&self) -> &str {
        self.input_devices
            .get(self.mic_idx)
            .map(|s| s.as_str())
            .unwrap_or("(nenhum)")
    }

    fn headphones_name(&self) -> &str {
        self.output_devices
            .get(self.headphones_idx)
            .map(|s| s.as_str())
            .unwrap_or("(nenhum)")
    }
}

// ─── Rendering ────────────────────────────────────────────────────────────────

impl eframe::App for SettingsApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Custom title bar + drag support
        let title_response = egui::TopBottomPanel::top("titlebar")
            .frame(egui::Frame::none().fill(BG_PRIMARY).inner_margin(egui::Margin {
                left: 16.0,
                right: 12.0,
                top: 10.0,
                bottom: 10.0,
            }))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    // Status dot
                    let dot_color = if self.is_active { ACCENT_GREEN } else { TEXT_MUTED };
                    let (dot_rect, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                    ui.painter().circle_filled(dot_rect.center(), 5.0, dot_color);

                    ui.add_space(8.0);

                    ui.label(
                        egui::RichText::new("Meeting Translator")
                            .color(TEXT_PRIMARY)
                            .size(15.0)
                            .strong(),
                    );

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        // Close button
                        let close_btn = ui.add(
                            egui::Button::new(egui::RichText::new("✕").color(TEXT_MUTED).size(13.0))
                                .fill(egui::Color32::TRANSPARENT)
                                .frame(false),
                        );
                        if close_btn.clicked() {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                        if close_btn.hovered() {
                            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                        }
                    });
                });
            });

        // Drag window by title bar
        let title_resp = title_response.response;
        if title_resp.is_pointer_button_down_on() {
            if self.drag_origin.is_none() {
                if let Some(pointer_pos) = ctx.pointer_interact_pos() {
                    if let Some(win_pos) = ctx.input(|i| i.viewport().outer_rect).map(|r| r.min) {
                        self.drag_origin = Some(pointer_pos);
                        self.window_pos = Some(win_pos);
                    }
                }
            }
            if let (Some(origin), Some(start_pos)) = (self.drag_origin, self.window_pos) {
                if let Some(current) = ctx.pointer_interact_pos() {
                    let delta = current - origin;
                    let new_pos = start_pos + delta;
                    ctx.send_viewport_cmd(egui::ViewportCommand::OuterPosition(new_pos));
                }
            }
        } else {
            self.drag_origin = None;
            self.window_pos = None;
        }

        // Main content
        egui::CentralPanel::default()
            .frame(egui::Frame::none().fill(BG_PRIMARY).inner_margin(egui::Margin::same(16.0)))
            .show(ctx, |ui| {
                ui.set_min_size(ui.available_size());

                // ── Microfone card ────────────────────────────────────────
                device_section(ui, "MICROFONE", |ui| {
                    // Device row
                    device_row(ui, "Dispositivo", |ui| {
                        let old = self.mic_idx;
                        let mic_display = self.mic_name().to_string();
                        device_combo(ui, "mic_device", &mic_display, &self.input_devices, &mut self.mic_idx);
                        if self.mic_idx != old {
                            let new_name = self.mic_name().to_string();
                            self.send(TrayAction::SetMicDevice(new_name));
                        }
                    });

                    ui.add_space(8.0);

                    // Language row
                    device_row(ui, "Idioma", |ui| {
                        let old = self.mic_source_lang;
                        language_selector(ui, "mic_lang", &mut self.mic_source_lang);
                        if self.mic_source_lang != old {
                            self.send(TrayAction::SetMicSourceLanguage(self.mic_source_lang));
                        }
                    });
                });

                ui.add_space(12.0);

                // ── Alto-falante card ─────────────────────────────────────
                device_section(ui, "ALTO-FALANTE", |ui| {
                    // Device row
                    device_row(ui, "Dispositivo", |ui| {
                        let old = self.headphones_idx;
                        let hp_display = self.headphones_name().to_string();
                        device_combo(ui, "fone_device", &hp_display, &self.output_devices, &mut self.headphones_idx);
                        if self.headphones_idx != old {
                            let new_name = self.headphones_name().to_string();
                            self.send(TrayAction::SetHeadphonesDevice(new_name));
                        }
                    });

                    ui.add_space(8.0);

                    // Language row
                    device_row(ui, "Idioma", |ui| {
                        let old = self.speaker_source_lang;
                        language_selector(ui, "spk_lang", &mut self.speaker_source_lang);
                        if self.speaker_source_lang != old {
                            self.send(TrayAction::SetSpeakerSourceLanguage(self.speaker_source_lang));
                        }
                    });
                });

                // Spacer to push button to bottom
                let remaining = ui.available_height() - 52.0;
                if remaining > 0.0 {
                    ui.add_space(remaining);
                }

                // ── Start / Stop button ───────────────────────────────────
                ui.with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
                    let (label, btn_color) = if self.is_active {
                        ("  ⏹  PARAR TRADUÇÃO  ", ACCENT_RED)
                    } else {
                        ("  ▶  INICIAR TRADUÇÃO  ", ACCENT_BLUE)
                    };

                    let btn = egui::Button::new(
                        egui::RichText::new(label)
                            .color(egui::Color32::WHITE)
                            .size(13.0)
                            .strong(),
                    )
                    .fill(btn_color)
                    .rounding(egui::Rounding::same(6.0))
                    .min_size(egui::vec2(200.0, 38.0));

                    if ui.add(btn).clicked() {
                        self.is_active = !self.is_active;
                        let cmd = if self.is_active {
                            PipelineCommand::Start
                        } else {
                            PipelineCommand::Stop
                        };
                        self.send(TrayAction::Command(cmd));
                    }
                });
            });
    }
}

// ─── Layout helpers ───────────────────────────────────────────────────────────

fn device_section(ui: &mut egui::Ui, title: &str, content: impl FnOnce(&mut egui::Ui)) {
    let card_frame = egui::Frame::none()
        .fill(BG_SECONDARY)
        .rounding(egui::Rounding::same(8.0))
        .inner_margin(egui::Margin::same(14.0));

    card_frame.show(ui, |ui| {
        // Section label
        ui.label(
            egui::RichText::new(title)
                .color(TEXT_HEADER)
                .size(10.5)
                .strong(),
        );
        ui.add_space(10.0);
        content(ui);
    });
}

fn device_row(ui: &mut egui::Ui, label: &str, content: impl FnOnce(&mut egui::Ui)) {
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(label)
                .color(TEXT_MUTED)
                .size(12.0),
        );
        // Align controls to right side
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), content);
    });
}

fn device_combo(
    ui: &mut egui::Ui,
    id: &str,
    selected_text: &str,
    items: &[String],
    idx: &mut usize,
) {
    // Truncate long device names
    let display = truncate_device_name(selected_text, 30);

    egui::ComboBox::from_id_salt(id)
        .width(240.0)
        .selected_text(display)
        .show_ui(ui, |ui| {
            for (i, name) in items.iter().enumerate() {
                let display_item = truncate_device_name(name, 40);
                ui.selectable_value(idx, i, display_item)
                    .on_hover_text(name);
            }
        });
}

fn language_selector(ui: &mut egui::Ui, id: &str, lang: &mut Language) {
    let target = opposite_lang(*lang);

    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(target.display_name())
                .color(TEXT_MUTED)
                .size(12.0),
        );
        ui.label(egui::RichText::new("←").color(ACCENT_ARROW).size(12.0));
        egui::ComboBox::from_id_salt(id)
            .width(100.0)
            .selected_text(lang.display_name())
            .show_ui(ui, |ui| {
                ui.selectable_value(lang, Language::Portuguese, "Português");
                ui.selectable_value(lang, Language::English, "English");
            });
    });
}

fn truncate_device_name(name: &str, max_chars: usize) -> String {
    if name.chars().count() <= max_chars {
        name.to_string()
    } else {
        let truncated: String = name.chars().take(max_chars - 1).collect();
        format!("{}…", truncated)
    }
}

fn opposite_lang(lang: Language) -> Language {
    match lang {
        Language::English => Language::Portuguese,
        Language::Portuguese => Language::English,
    }
}

// ─── Theme ────────────────────────────────────────────────────────────────────

fn configure_style(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();

    style.spacing.item_spacing = egui::vec2(8.0, 4.0);
    style.spacing.button_padding = egui::vec2(12.0, 6.0);
    style.spacing.combo_width = 240.0;

    // Visuals
    let mut visuals = egui::Visuals::dark();
    visuals.window_rounding = egui::Rounding::same(0.0);
    visuals.window_fill = BG_PRIMARY;
    visuals.panel_fill = BG_PRIMARY;
    visuals.override_text_color = Some(TEXT_PRIMARY);

    // Widgets
    visuals.widgets.inactive.bg_fill = BG_INPUT;
    visuals.widgets.inactive.rounding = egui::Rounding::same(4.0);
    visuals.widgets.inactive.fg_stroke = egui::Stroke::new(1.0, TEXT_MUTED);

    visuals.widgets.hovered.bg_fill = BG_CARD;
    visuals.widgets.hovered.rounding = egui::Rounding::same(4.0);
    visuals.widgets.hovered.fg_stroke = egui::Stroke::new(1.0, TEXT_PRIMARY);

    visuals.widgets.active.bg_fill = ACCENT_BLUE;
    visuals.widgets.active.rounding = egui::Rounding::same(4.0);
    visuals.widgets.active.fg_stroke = egui::Stroke::new(1.0, egui::Color32::WHITE);

    visuals.widgets.open.bg_fill = BG_INPUT;
    visuals.widgets.open.rounding = egui::Rounding::same(4.0);

    visuals.selection.bg_fill = egui::Color32::from_rgba_premultiplied(88, 101, 242, 60);

    // Popup (dropdown) background
    visuals.popup_shadow = egui::epaint::Shadow {
        offset: egui::vec2(0.0, 4.0),
        blur: 12.0,
        spread: 0.0,
        color: egui::Color32::from_black_alpha(80),
    };

    style.visuals = visuals;
    ctx.set_style(style);

    // Larger base font
    let mut fonts = egui::FontDefinitions::default();
    fonts
        .families
        .entry(egui::FontFamily::Proportional)
        .or_default();
    ctx.set_fonts(fonts);
}
