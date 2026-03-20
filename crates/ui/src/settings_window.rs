use eframe::egui;
use shared::{Language, PipelineCommand};
use std::sync::mpsc as std_mpsc;
use std::sync::{Arc, Mutex};

use crate::TrayAction;

// ─── Win32 helpers for reliable show/hide ────────────────────────────────────

/// Unique window title so FindWindowW can locate our HWND reliably.
const WINDOW_TITLE: &str = "Meeting Translator Settings";

#[cfg(windows)]
mod win32 {
    use windows_sys::Win32::Foundation::HWND;
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        FindWindowW, SetForegroundWindow, ShowWindow, SW_HIDE, SW_SHOW,
    };

    /// Find our settings window by its exact title.
    pub fn find_hwnd() -> Option<HWND> {
        let title: Vec<u16> = super::WINDOW_TITLE
            .encode_utf16()
            .chain(std::iter::once(0))
            .collect();
        let hwnd = unsafe { FindWindowW(std::ptr::null(), title.as_ptr()) };
        if hwnd.is_null() {
            None
        } else {
            Some(hwnd)
        }
    }

    pub fn hide_window() {
        if let Some(hwnd) = find_hwnd() {
            unsafe { ShowWindow(hwnd, SW_HIDE) };
        }
    }

    pub fn show_window() {
        if let Some(hwnd) = find_hwnd() {
            unsafe {
                ShowWindow(hwnd, SW_SHOW);
                SetForegroundWindow(hwnd);
            };
        }
    }
}

// ─── shadcn/ui zinc dark palette ─────────────────────────────────────────────

const BG_BASE: egui::Color32 = egui::Color32::from_rgb(9, 9, 11);         // zinc-950
const BG_CARD: egui::Color32 = egui::Color32::from_rgb(24, 24, 27);       // zinc-900
const BG_INPUT: egui::Color32 = egui::Color32::from_rgb(9, 9, 11);        // zinc-950
const BORDER: egui::Color32 = egui::Color32::from_rgb(39, 39, 42);        // zinc-800
const BORDER_HOVER: egui::Color32 = egui::Color32::from_rgb(82, 82, 91);  // zinc-600
const TEXT_PRIMARY: egui::Color32 = egui::Color32::from_rgb(250, 250, 250); // zinc-50
const TEXT_MUTED: egui::Color32 = egui::Color32::from_rgb(161, 161, 170);  // zinc-400
const TEXT_SUBTLE: egui::Color32 = egui::Color32::from_rgb(113, 113, 122); // zinc-500
const ACCENT_INDIGO: egui::Color32 = egui::Color32::from_rgb(99, 102, 241); // indigo-500
const ACCENT_GREEN: egui::Color32 = egui::Color32::from_rgb(34, 197, 94);   // green-500
const ACCENT_RED: egui::Color32 = egui::Color32::from_rgb(239, 68, 68);     // red-500

// ─── Init data ────────────────────────────────────────────────────────────────

pub struct SettingsInit {
    pub output_devices: Vec<String>,
    pub input_devices: Vec<String>,
    pub selected_mic: String,
    pub selected_headphones: String,
    pub mic_source_lang: Language,
    pub speaker_source_lang: Language,
    pub is_active: bool,
}

// ─── Window entry point ───────────────────────────────────────────────────────

pub fn open(
    init: SettingsInit,
    action_tx: std_mpsc::Sender<TrayAction>,
    show_rx: std_mpsc::Receiver<SettingsInit>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let pending_show: Arc<Mutex<Option<SettingsInit>>> = Arc::new(Mutex::new(None));

        let options = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([460.0, 440.0])
                .with_min_inner_size([420.0, 400.0])
                .with_resizable(true)
                .with_always_on_top(),
            event_loop_builder: Some(Box::new(|builder| {
                use winit::platform::windows::EventLoopBuilderExtWindows;
                builder.with_any_thread(true);
            })),
            ..Default::default()
        };

        let tx = action_tx;
        let pending = Arc::clone(&pending_show);
        let _ = eframe::run_native(
            WINDOW_TITLE,
            options,
            Box::new(move |cc| {
                configure_style(&cc.egui_ctx);

                // Spawn a watcher thread that blocks on the show channel.
                // When a show signal arrives, it stores the data, makes the
                // window visible via Win32 API, and wakes the event loop.
                let ctx = cc.egui_ctx.clone();
                let pending_for_watcher = Arc::clone(&pending);
                std::thread::spawn(move || {
                    while let Ok(init) = show_rx.recv() {
                        *pending_for_watcher.lock().unwrap() = Some(init);
                        #[cfg(windows)]
                        win32::show_window();
                        ctx.request_repaint();
                    }
                });

                Ok(Box::new(SettingsApp::new(init, tx, pending)))
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
    pending_show: Arc<Mutex<Option<SettingsInit>>>,
}

impl SettingsApp {
    fn new(
        init: SettingsInit,
        action_tx: std_mpsc::Sender<TrayAction>,
        pending_show: Arc<Mutex<Option<SettingsInit>>>,
    ) -> Self {
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
            pending_show,
        }
    }

    fn apply_init(&mut self, init: SettingsInit) {
        self.mic_idx = init
            .input_devices
            .iter()
            .position(|d| *d == init.selected_mic)
            .unwrap_or(0);
        self.headphones_idx = init
            .output_devices
            .iter()
            .position(|d| *d == init.selected_headphones)
            .unwrap_or(0);
        self.output_devices = init.output_devices;
        self.input_devices = init.input_devices;
        self.mic_source_lang = init.mic_source_lang;
        self.speaker_source_lang = init.speaker_source_lang;
        self.is_active = init.is_active;
    }

    fn send(&self, action: TrayAction) {
        let _ = self.action_tx.send(action);
    }

    fn mic_name(&self) -> String {
        self.input_devices
            .get(self.mic_idx)
            .cloned()
            .unwrap_or_else(|| "(nenhum)".to_string())
    }

    fn headphones_name(&self) -> String {
        self.output_devices
            .get(self.headphones_idx)
            .cloned()
            .unwrap_or_else(|| "(nenhum)".to_string())
    }
}

// ─── Rendering ────────────────────────────────────────────────────────────────

impl eframe::App for SettingsApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Check for pending "show" signal set by the watcher thread
        let pending_init = self.pending_show.lock().unwrap().take();
        if let Some(init) = pending_init {
            self.apply_init(init);
        }

        // Intercept close: hide the window via Win32 instead of destroying it.
        // CancelClose keeps the event loop alive; SW_HIDE removes from screen.
        if ctx.input(|i| i.viewport().close_requested()) {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            #[cfg(windows)]
            win32::hide_window();
        }

        // Status bar
        egui::TopBottomPanel::top("status_bar")
            .frame(
                egui::Frame::none()
                    .fill(BG_BASE)
                    .inner_margin(egui::Margin { left: 16.0, right: 16.0, top: 8.0, bottom: 8.0 }),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    let dot_color = if self.is_active { ACCENT_GREEN } else { TEXT_SUBTLE };
                    let (dot_rect, _) =
                        ui.allocate_exact_size(egui::vec2(8.0, 8.0), egui::Sense::hover());
                    ui.painter().circle_filled(dot_rect.center(), 4.0, dot_color);
                    ui.add_space(6.0);
                    let (status, color) = if self.is_active {
                        ("Tradução ativa", ACCENT_GREEN)
                    } else {
                        ("Tradução pausada", TEXT_SUBTLE)
                    };
                    ui.label(egui::RichText::new(status).color(color).size(11.5));
                });
            });

        // Action button pinned at bottom
        egui::TopBottomPanel::bottom("action_bar")
            .frame(
                egui::Frame::none()
                    .fill(BG_BASE)
                    .inner_margin(egui::Margin::same(16.0)),
            )
            .show(ctx, |ui| {
                let content_width = ui.available_width();
                let (label, btn_color) = if self.is_active {
                    ("⏹   Parar tradução", ACCENT_RED)
                } else {
                    ("▶   Iniciar tradução", ACCENT_INDIGO)
                };

                let btn = egui::Button::new(
                    egui::RichText::new(label)
                        .color(egui::Color32::WHITE)
                        .size(13.5)
                        .strong(),
                )
                .fill(btn_color)
                .rounding(egui::Rounding::same(6.0))
                .min_size(egui::vec2(content_width, 42.0));

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

        // Scrollable content
        egui::CentralPanel::default()
            .frame(
                egui::Frame::none()
                    .fill(BG_BASE)
                    .inner_margin(egui::Margin::same(16.0)),
            )
            .show(ctx, |ui| {
                let content_width = ui.available_width();

                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.set_width(content_width);

                        // ── Microfone card ─────────────────────────────────
                        shadcn_card(ui, content_width, |ui| {
                            card_header(ui, "🎙  Microfone");

                            field_label(ui, "Dispositivo de entrada");
                            {
                                let old = self.mic_idx;
                                let name = self.mic_name();
                                full_width_combo(ui, "mic_device", &name, &self.input_devices, &mut self.mic_idx, content_width);
                                if self.mic_idx != old {
                                    self.send(TrayAction::SetMicDevice(self.mic_name()));
                                }
                            }

                            ui.add_space(12.0);

                            field_label(ui, "Idioma");
                            {
                                let old = self.mic_source_lang;
                                language_field(ui, "mic_lang", "Eu falo em", "sai em", &mut self.mic_source_lang);
                                if self.mic_source_lang != old {
                                    self.send(TrayAction::SetMicSourceLanguage(self.mic_source_lang));
                                }
                            }
                        });

                        ui.add_space(12.0);

                        // ── Alto-falante card ───────────────────────────────
                        shadcn_card(ui, content_width, |ui| {
                            card_header(ui, "🔊  Alto-falante");

                            field_label(ui, "Dispositivo de saída");
                            {
                                let old = self.headphones_idx;
                                let name = self.headphones_name();
                                full_width_combo(ui, "fone_device", &name, &self.output_devices, &mut self.headphones_idx, content_width);
                                if self.headphones_idx != old {
                                    self.send(TrayAction::SetHeadphonesDevice(self.headphones_name()));
                                }
                            }

                            ui.add_space(12.0);

                            field_label(ui, "Idioma");
                            {
                                let old = self.speaker_source_lang;
                                language_field(ui, "spk_lang", "Reunião em", "eu escuto em", &mut self.speaker_source_lang);
                                if self.speaker_source_lang != old {
                                    self.send(TrayAction::SetSpeakerSourceLanguage(self.speaker_source_lang));
                                }
                            }
                        });

                        ui.add_space(8.0);
                    });
            });
    }
}

// ─── shadcn-style component helpers ──────────────────────────────────────────

fn shadcn_card(ui: &mut egui::Ui, width: f32, content: impl FnOnce(&mut egui::Ui)) {
    let frame = egui::Frame::none()
        .fill(BG_CARD)
        .stroke(egui::Stroke::new(1.0, BORDER))
        .rounding(egui::Rounding::same(8.0))
        .inner_margin(egui::Margin::same(16.0));

    frame.show(ui, |ui| {
        // Force card to fill available width
        ui.set_width(width - 32.0);
        content(ui);
    });
}

fn card_header(ui: &mut egui::Ui, title: &str) {
    ui.label(
        egui::RichText::new(title)
            .color(TEXT_PRIMARY)
            .size(13.0)
            .strong(),
    );

    ui.add(egui::Separator::default().spacing(10.0).shrink(0.0));
    ui.add_space(2.0);
}

fn field_label(ui: &mut egui::Ui, label: &str) {
    ui.label(
        egui::RichText::new(label)
            .color(TEXT_MUTED)
            .size(11.5),
    );
    ui.add_space(4.0);
}

fn full_width_combo(
    ui: &mut egui::Ui,
    id: &str,
    selected: &str,
    items: &[String],
    idx: &mut usize,
    card_width: f32,
) {
    // Subtract outer margin (16*2) + card inner margin (16*2) + stroke (1*2) + a little extra
    let combo_width = card_width - 32.0 - 32.0 - 4.0;
    let display = truncate(selected, 46);

    egui::ComboBox::from_id_salt(id)
        .width(combo_width.max(200.0))
        .selected_text(display)
        .show_ui(ui, |ui| {
            for (i, name) in items.iter().enumerate() {
                ui.selectable_value(idx, i, truncate(name, 55))
                    .on_hover_text(name);
            }
        });
}

/// `context_label`: e.g. "Eu falo em" | `connector`: e.g. "sai em" or "eu escuto em"
fn language_field(
    ui: &mut egui::Ui,
    id: &str,
    context_label: &str,
    connector: &str,
    lang: &mut Language,
) {
    let target = opposite_lang(*lang);

    // "Eu falo em  [Português ▼]  sai em  English"
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(context_label).color(TEXT_MUTED).size(12.0));

        ui.add_space(4.0);

        egui::ComboBox::from_id_salt(id)
            .width(115.0)
            .selected_text(lang.display_name())
            .show_ui(ui, |ui| {
                ui.selectable_value(lang, Language::Portuguese, "Português");
                ui.selectable_value(lang, Language::English, "English");
            });

        ui.add_space(6.0);
        ui.label(egui::RichText::new(connector).color(TEXT_SUBTLE).size(12.0));
        ui.add_space(4.0);
        ui.label(egui::RichText::new(target.display_name()).color(TEXT_PRIMARY).size(12.0).strong());
    });
}

fn opposite_lang(lang: Language) -> Language {
    match lang {
        Language::English => Language::Portuguese,
        Language::Portuguese => Language::English,
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(max - 1).collect::<String>())
    }
}

// ─── Global theme ─────────────────────────────────────────────────────────────

fn configure_style(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();

    style.spacing.item_spacing = egui::vec2(8.0, 5.0);
    style.spacing.button_padding = egui::vec2(14.0, 8.0);
    style.spacing.combo_width = 200.0;
    style.spacing.interact_size.y = 32.0;

    let mut visuals = egui::Visuals::dark();
    visuals.window_fill = BG_BASE;
    visuals.panel_fill = BG_BASE;
    visuals.override_text_color = Some(TEXT_PRIMARY);
    visuals.window_rounding = egui::Rounding::same(0.0);

    // Default (inactive) widgets — inputs, combos
    visuals.widgets.inactive.bg_fill = BG_INPUT;
    visuals.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, BORDER);
    visuals.widgets.inactive.rounding = egui::Rounding::same(6.0);
    visuals.widgets.inactive.fg_stroke = egui::Stroke::new(1.0, TEXT_MUTED);

    // Hovered
    visuals.widgets.hovered.bg_fill = BG_CARD;
    visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, BORDER_HOVER);
    visuals.widgets.hovered.rounding = egui::Rounding::same(6.0);
    visuals.widgets.hovered.fg_stroke = egui::Stroke::new(1.0, TEXT_PRIMARY);

    // Active (pressed)
    visuals.widgets.active.bg_fill = ACCENT_INDIGO;
    visuals.widgets.active.bg_stroke = egui::Stroke::NONE;
    visuals.widgets.active.rounding = egui::Rounding::same(6.0);
    visuals.widgets.active.fg_stroke = egui::Stroke::new(1.0, egui::Color32::WHITE);

    // Open (combo open state)
    visuals.widgets.open.bg_fill = BG_INPUT;
    visuals.widgets.open.bg_stroke = egui::Stroke::new(1.0, ACCENT_INDIGO);
    visuals.widgets.open.rounding = egui::Rounding::same(6.0);

    // Selection highlight
    visuals.selection.bg_fill =
        egui::Color32::from_rgba_premultiplied(99, 102, 241, 50);

    // Separator color
    visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, BORDER);

    // Popup shadow
    visuals.popup_shadow = egui::epaint::Shadow {
        offset: egui::vec2(0.0, 8.0),
        blur: 20.0,
        spread: 0.0,
        color: egui::Color32::from_black_alpha(120),
    };

    style.visuals = visuals;
    ctx.set_style(style);
}
