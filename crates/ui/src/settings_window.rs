use eframe::egui;
use shared::{Language, PipelineCommand};
use std::sync::mpsc as std_mpsc;

use crate::TrayAction;

/// Data needed to populate the settings window controls.
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

/// Open the settings window in a new thread. Returns the thread handle.
pub fn open(
    init: SettingsInit,
    action_tx: std_mpsc::Sender<TrayAction>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let options = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([560.0, 130.0])
                .with_resizable(false)
                .with_always_on_top(),
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

// ─── App state ───────────────────────────────────────────────────────────────

struct SettingsApp {
    output_devices: Vec<String>,
    input_devices: Vec<String>,
    mic_idx: usize,
    headphones_idx: usize,
    /// Language the user speaks (mic pipeline source). Target is always opposite.
    mic_source_lang: Language,
    /// Language the incoming audio is in (speaker/loopback pipeline source). Target is always opposite.
    speaker_source_lang: Language,
    is_active: bool,
    action_tx: std_mpsc::Sender<TrayAction>,
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

// ─── UI rendering ─────────────────────────────────────────────────────────────

impl eframe::App for SettingsApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add_space(6.0);

            egui::Grid::new("main_grid")
                .num_columns(4)
                .spacing([12.0, 10.0])
                .min_col_width(50.0)
                .show(ui, |ui| {
                    // ── Row 1: Mic ─────────────────────────────────────────
                    ui.label("Mic:");

                    let old_mic = self.mic_idx;
                    egui::ComboBox::from_id_salt("mic")
                        .width(220.0)
                        .selected_text(self.mic_name())
                        .show_ui(ui, |ui| {
                            for (i, name) in self.input_devices.iter().enumerate() {
                                ui.selectable_value(&mut self.mic_idx, i, name);
                            }
                        });
                    if self.mic_idx != old_mic {
                        self.send(TrayAction::SetMicDevice(self.mic_name().to_string()));
                    }

                    // Language: source ComboBox → target label (computed as opposite)
                    let old_mic_src = self.mic_source_lang;
                    let mic_target_name = opposite_lang(self.mic_source_lang).display_name();
                    ui.horizontal(|ui| {
                        language_combo(ui, "mic_src", &mut self.mic_source_lang);
                        ui.label("→");
                        ui.label(
                            egui::RichText::new(mic_target_name).color(egui::Color32::GRAY),
                        );
                    });
                    if self.mic_source_lang != old_mic_src {
                        // Mic direction changed independently — Fone is NOT affected
                        self.send(TrayAction::SetMicSourceLanguage(self.mic_source_lang));
                    }

                    // Active toggle
                    let toggle_changed = toggle_switch(ui, &mut self.is_active);
                    if toggle_changed {
                        let cmd = if self.is_active {
                            PipelineCommand::Start
                        } else {
                            PipelineCommand::Stop
                        };
                        self.send(TrayAction::Command(cmd));
                    }
                    ui.end_row();

                    // ── Row 2: Fone (Headphones) ───────────────────────────
                    // Loopback is automatically the same device as the headphones,
                    // so the user only selects one output device here.
                    ui.label("Fone:");

                    let old_hp = self.headphones_idx;
                    egui::ComboBox::from_id_salt("fone")
                        .width(220.0)
                        .selected_text(self.headphones_name())
                        .show_ui(ui, |ui| {
                            for (i, name) in self.output_devices.iter().enumerate() {
                                ui.selectable_value(&mut self.headphones_idx, i, name);
                            }
                        });
                    if self.headphones_idx != old_hp {
                        // Changing the headphones also implicitly updates the loopback
                        // capture device — handled in main.rs via SetHeadphonesDevice.
                        self.send(TrayAction::SetHeadphonesDevice(
                            self.headphones_name().to_string(),
                        ));
                    }

                    // Language: source ComboBox → target label (computed as opposite)
                    let old_spk_src = self.speaker_source_lang;
                    let spk_target_name = opposite_lang(self.speaker_source_lang).display_name();
                    ui.horizontal(|ui| {
                        language_combo(ui, "spk_src", &mut self.speaker_source_lang);
                        ui.label("→");
                        ui.label(
                            egui::RichText::new(spk_target_name).color(egui::Color32::GRAY),
                        );
                    });
                    if self.speaker_source_lang != old_spk_src {
                        // Fone direction changed independently — Mic is NOT affected
                        self.send(TrayAction::SetSpeakerSourceLanguage(self.speaker_source_lang));
                    }

                    ui.end_row();
                });
        });
    }
}

// ─── Language helpers ──────────────────────────────────────────────────────────

fn language_combo(ui: &mut egui::Ui, id: &str, lang: &mut Language) {
    egui::ComboBox::from_id_salt(id)
        .width(90.0)
        .selected_text(lang.display_name())
        .show_ui(ui, |ui| {
            ui.selectable_value(lang, Language::Portuguese, "Português");
            ui.selectable_value(lang, Language::English, "English");
        });
}

fn opposite_lang(lang: Language) -> Language {
    match lang {
        Language::English => Language::Portuguese,
        Language::Portuguese => Language::English,
    }
}

// ─── Custom toggle switch widget ──────────────────────────────────────────────

fn toggle_switch(ui: &mut egui::Ui, on: &mut bool) -> bool {
    let desired_size = egui::vec2(44.0, 22.0);
    let (rect, response) = ui.allocate_exact_size(desired_size, egui::Sense::click());

    let mut changed = false;
    if response.clicked() {
        *on = !*on;
        changed = true;
    }

    if ui.is_rect_visible(rect) {
        let how_on = ui.ctx().animate_bool_with_time(response.id, *on, 0.15);
        let radius = 0.5 * rect.height();

        let bg_color = if *on {
            egui::Color32::from_rgb(76, 175, 80)
        } else {
            egui::Color32::from_rgb(158, 158, 158)
        };
        ui.painter().rect(rect, radius, bg_color, egui::Stroke::NONE);

        let circle_x = egui::lerp((rect.left() + radius)..=(rect.right() - radius), how_on);
        let center = egui::pos2(circle_x, rect.center().y);
        ui.painter().circle(
            center,
            radius - 2.0,
            egui::Color32::WHITE,
            egui::Stroke::NONE,
        );
    }

    changed
}

// ─── Theme ────────────────────────────────────────────────────────────────────

fn configure_style(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = egui::vec2(8.0, 6.0);
    style.visuals.window_rounding = egui::Rounding::same(10.0);
    ctx.set_style(style);
}
