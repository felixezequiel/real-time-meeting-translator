use audio::recorder::VoiceRecorder;
use eframe::egui;
use shared::{Language, PipelineCommand, StageMetricsAggregator, StageStats};
use std::path::PathBuf;
use std::sync::mpsc as std_mpsc;
use std::sync::{Arc, Mutex};

use crate::TrayAction;

/// Calibration prompts, ordered short → long. Reading all three back
/// to back gives OpenVoice's SE extractor a varied ~30 s sample (the
/// extractor is most stable above ~15 s of voiced speech, and longer
/// recordings smooth out per-phoneme energy quirks). Texts cover
/// open vowels, plosives, sibilants, and a couple of question
/// intonations to capture pitch dynamics.
const CALIBRATION_PROMPTS: &[&str] = &[
    "O sol da manhã ilumina a sala enquanto eu preparo um café forte.",
    "Hoje é um bom dia para revisar minhas anotações, ajustar o foco e seguir com o trabalho — vamos ver até onde chegamos antes do almoço.",
    "Quando alguém me pergunta sobre o futuro, costumo responder com calma: planejo os próximos meses passo a passo, escolho as prioridades com cuidado e tento manter um equilíbrio saudável entre prazo e qualidade. E você, como organiza suas semanas?",
];

/// Minimum recording length required before we'll accept the save.
/// Below ~15 s the SE extractor produces noisy embeddings.
const MIN_RECORDING_SECONDS: f32 = 15.0;

/// Stages displayed in the metrics panel, in pipeline order. The first
/// element is the internal stage key emitted by the pipeline; the second
/// is the human-readable label shown in the UI. Labels were renamed
/// from technical jargon (VAD/STT/MT/TTS) to plain Portuguese so the
/// panel makes sense to anyone, not just to people who built the
/// pipeline. The mapping table in plain English for posterity:
///   vad                       → "Detecção de fala"
///   stt                       → "Reconhecimento de voz"
///   translate_first_fragment  → "Tempo até 1ª palavra"
///   translate                 → "Tradução completa"
///   tts                       → "Geração de áudio"
///   total                     → "Latência ponta-a-ponta"
const METRIC_STAGES: &[(&str, &str)] = &[
    ("vad",                      "Detecção de fala"),
    ("stt",                      "Reconhecimento de voz"),
    ("translate_first_fragment", "Tempo até 1ª palavra"),
    ("translate",                "Tradução completa"),
    ("tts",                      "Geração de áudio"),
    ("ttfa",                     "Tempo até 1º som"),
    ("total",                    "Latência ponta-a-ponta"),
];

/// How often the settings window is forced to repaint while open. The
/// metrics panel reads the aggregator on each frame, so without this
/// the numbers would only refresh when the window receives focus or
/// the user moves the mouse over it.
const METRICS_REPAINT_INTERVAL_MS: u64 = 500;

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
    /// Path to the user's currently saved voice profile WAV, if any.
    /// `None` when no profile has been recorded yet.
    pub voice_profile_path: Option<String>,
    /// Directory under which new profile recordings are written
    /// (typically `<app_dir>/voice_profile/`).
    pub voice_profile_dir: std::path::PathBuf,
}

// ─── Window entry point ───────────────────────────────────────────────────────

pub fn open(
    init: SettingsInit,
    action_tx: std_mpsc::Sender<TrayAction>,
    show_rx: std_mpsc::Receiver<SettingsInit>,
    metrics: Arc<StageMetricsAggregator>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let pending_show: Arc<Mutex<Option<SettingsInit>>> = Arc::new(Mutex::new(None));

        let options = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([460.0, 540.0])
                .with_min_inner_size([420.0, 500.0])
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

                Ok(Box::new(SettingsApp::new(init, tx, pending, metrics)))
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
    metrics: Arc<StageMetricsAggregator>,
    /// Voice profile state — see VoiceProfileState for the modes.
    voice_profile: VoiceProfileState,
    voice_profile_dir: PathBuf,
}

/// Tracks the voice-profile recording flow. Three modes:
/// - `Idle`: showing the profile status + a "Gravar minha voz" button.
/// - `Recording`: active recorder, level meter visible.
/// - `JustSaved`: confirmation banner that decays back to Idle on the
///   next render cycle (we keep the path for one frame so the UI can
///   show "Perfil salvo" briefly).
enum VoiceProfileState {
    Idle {
        saved_path: Option<String>,
        last_error: Option<String>,
    },
    Recording {
        recorder: VoiceRecorder,
        started_at: std::time::Instant,
    },
}

impl SettingsApp {
    fn new(
        init: SettingsInit,
        action_tx: std_mpsc::Sender<TrayAction>,
        pending_show: Arc<Mutex<Option<SettingsInit>>>,
        metrics: Arc<StageMetricsAggregator>,
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
            metrics,
            voice_profile: VoiceProfileState::Idle {
                saved_path: init.voice_profile_path,
                last_error: None,
            },
            voice_profile_dir: init.voice_profile_dir,
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
        // Don't clobber an in-flight recording; only refresh when idle.
        if matches!(self.voice_profile, VoiceProfileState::Idle { .. }) {
            self.voice_profile = VoiceProfileState::Idle {
                saved_path: init.voice_profile_path,
                last_error: None,
            };
        }
        self.voice_profile_dir = init.voice_profile_dir;
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

                            ui.add_space(12.0);

                            field_label(ui, "Perfil de voz");
                            self.render_voice_profile(ui);
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

                        ui.add_space(12.0);

                        // ── Métricas card ───────────────────────────────────
                        let snapshot = self.metrics.snapshot();
                        shadcn_card(ui, content_width, |ui| {
                            card_header(ui, "📊  Latência por estágio");
                            metrics_table(ui, &snapshot);
                        });

                        ui.add_space(8.0);
                    });
            });

        // Schedule a periodic repaint while open so the metrics panel
        // refreshes without user interaction. egui only repaints on
        // events otherwise, which would freeze the numbers between
        // mouse moves.
        ctx.request_repaint_after(std::time::Duration::from_millis(
            METRICS_REPAINT_INTERVAL_MS,
        ));
    }
}

impl SettingsApp {
    fn render_voice_profile(&mut self, ui: &mut egui::Ui) {
        // Take ownership so we can transition state without fighting
        // the borrow checker.
        let current = std::mem::replace(
            &mut self.voice_profile,
            VoiceProfileState::Idle {
                saved_path: None,
                last_error: None,
            },
        );

        let next = match current {
            VoiceProfileState::Idle { saved_path, last_error } => {
                self.render_idle_voice_profile(ui, saved_path, last_error)
            }
            VoiceProfileState::Recording { recorder, started_at } => {
                self.render_recording_voice_profile(ui, recorder, started_at)
            }
        };
        self.voice_profile = next;
    }

    fn render_idle_voice_profile(
        &mut self,
        ui: &mut egui::Ui,
        saved_path: Option<String>,
        last_error: Option<String>,
    ) -> VoiceProfileState {
        let mut next_saved = saved_path.clone();
        let mut next_error = last_error;

        // Status line: "Perfil salvo" or "Sem perfil gravado".
        ui.horizontal(|ui| {
            let (label, color) = match &saved_path {
                Some(_) => ("✓ Perfil salvo", ACCENT_GREEN),
                None => ("Sem perfil gravado", TEXT_SUBTLE),
            };
            ui.label(egui::RichText::new(label).color(color).size(12.0));
        });

        if let Some(err) = &next_error {
            ui.add_space(2.0);
            ui.label(
                egui::RichText::new(err)
                    .color(ACCENT_RED)
                    .size(11.0),
            );
        }

        ui.add_space(6.0);

        // Help text reminding the user how the recording works.
        ui.label(
            egui::RichText::new(
                "Leia em voz alta as 3 frases abaixo, na ordem. \
                 Quanto mais natural a entonação, melhor a clonagem.",
            )
            .color(TEXT_MUTED)
            .size(11.0),
        );
        ui.add_space(4.0);
        for (i, prompt) in CALIBRATION_PROMPTS.iter().enumerate() {
            ui.label(
                egui::RichText::new(format!("{}. {}", i + 1, prompt))
                    .color(TEXT_PRIMARY)
                    .size(11.5),
            );
            ui.add_space(2.0);
        }

        ui.add_space(8.0);

        // Action buttons.
        ui.horizontal(|ui| {
            let label = if saved_path.is_some() {
                "🔁  Regravar voz"
            } else {
                "🎙  Gravar minha voz"
            };
            let btn = egui::Button::new(
                egui::RichText::new(label)
                    .color(egui::Color32::WHITE)
                    .size(12.0),
            )
            .fill(ACCENT_INDIGO)
            .rounding(egui::Rounding::same(6.0));

            // Disable while the pipelines are running — recording from
            // the same mic at the same time as the live pipeline would
            // contend for the cpal device on Windows.
            let enabled = !self.is_active;
            if ui.add_enabled(enabled, btn).clicked() {
                match self.start_recording() {
                    Ok(state) => return Some(state),
                    Err(msg) => next_error = Some(msg),
                }
            }

            if saved_path.is_some() {
                ui.add_space(8.0);
                let clear_btn = egui::Button::new(
                    egui::RichText::new("✕  Limpar perfil")
                        .color(TEXT_PRIMARY)
                        .size(12.0),
                )
                .fill(BG_INPUT)
                .stroke(egui::Stroke::new(1.0, BORDER))
                .rounding(egui::Rounding::same(6.0));
                if ui.add_enabled(enabled, clear_btn).clicked() {
                    next_saved = None;
                    self.send(TrayAction::SetVoiceProfile(None));
                }
            }
            None::<VoiceProfileState>
        })
        .inner
        .unwrap_or_else(|| VoiceProfileState::Idle {
            saved_path: next_saved,
            last_error: next_error,
        })
    }

    fn render_recording_voice_profile(
        &mut self,
        ui: &mut egui::Ui,
        recorder: VoiceRecorder,
        started_at: std::time::Instant,
    ) -> VoiceProfileState {
        let elapsed = started_at.elapsed().as_secs_f32();
        let rms = recorder.current_rms();

        // Big status line + timer.
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new("● Gravando")
                    .color(ACCENT_RED)
                    .size(13.0)
                    .strong(),
            );
            ui.add_space(8.0);
            ui.label(
                egui::RichText::new(format!("{:>5.1} s", elapsed))
                    .color(TEXT_PRIMARY)
                    .size(12.0),
            );
            ui.add_space(8.0);
            let remaining = (MIN_RECORDING_SECONDS - elapsed).max(0.0);
            if remaining > 0.0 {
                ui.label(
                    egui::RichText::new(format!("(mín. mais {:.0} s)", remaining))
                        .color(TEXT_SUBTLE)
                        .size(11.0),
                );
            } else {
                ui.label(
                    egui::RichText::new("✓ duração suficiente")
                        .color(ACCENT_GREEN)
                        .size(11.0),
                );
            }
        });

        // No-audio warning: if 2 s in the buffer is still empty, the
        // device opened but cpal never delivered samples (wrong
        // device, muted hardware). The check is on `sample_count`
        // (monotonic — only grows) instead of RMS (which oscillates
        // with speech), so once audio starts arriving the message
        // disappears for good and doesn't flicker between syllables.
        if elapsed > 2.0 && recorder.sample_count() == 0 {
            ui.add_space(2.0);
            ui.label(
                egui::RichText::new(
                    "⚠ Nenhum áudio recebido. Verifique se o dispositivo \
                     selecionado é o microfone real (e não um cable virtual).",
                )
                .color(ACCENT_RED)
                .size(11.0),
            );
        }

        ui.add_space(6.0);

        // Level meter — log-ish bar from 0..1, RMS values typically
        // peak around 0.05–0.20 for normal speech.
        let bar_height = 8.0;
        let bar_width = ui.available_width() - 8.0;
        let (rect, _) = ui.allocate_exact_size(
            egui::vec2(bar_width, bar_height),
            egui::Sense::hover(),
        );
        ui.painter().rect_filled(
            rect,
            egui::Rounding::same(2.0),
            BG_INPUT,
        );
        let level = (rms * 6.0).clamp(0.0, 1.0);
        let fill_width = bar_width * level;
        let fill_rect = egui::Rect::from_min_size(
            rect.min,
            egui::vec2(fill_width, bar_height),
        );
        let fill_color = if level > 0.85 {
            ACCENT_RED
        } else if level > 0.05 {
            ACCENT_GREEN
        } else {
            TEXT_SUBTLE
        };
        ui.painter().rect_filled(
            fill_rect,
            egui::Rounding::same(2.0),
            fill_color,
        );

        ui.add_space(8.0);

        // Show prompts again as a reading aid.
        for (i, prompt) in CALIBRATION_PROMPTS.iter().enumerate() {
            ui.label(
                egui::RichText::new(format!("{}. {}", i + 1, prompt))
                    .color(TEXT_PRIMARY)
                    .size(11.5),
            );
            ui.add_space(2.0);
        }

        ui.add_space(8.0);

        // Stop & save button — disabled until we've reached the
        // minimum length so the user doesn't accidentally submit a
        // 2-second clip. Both the wall-clock and the captured
        // duration must clear the threshold; otherwise a stuck
        // device that opened but produces no samples would still
        // let the green button appear.
        let captured_seconds = recorder.duration_seconds();
        let can_save = elapsed >= MIN_RECORDING_SECONDS
            && captured_seconds >= MIN_RECORDING_SECONDS;
        let save_btn = egui::Button::new(
            egui::RichText::new("⏹  Parar e salvar")
                .color(egui::Color32::WHITE)
                .size(12.0),
        )
        .fill(if can_save { ACCENT_GREEN } else { TEXT_SUBTLE })
        .rounding(egui::Rounding::same(6.0));

        // We own `recorder` for the duration of this function. The
        // save/cancel branches consume it; everything else returns it
        // back into the Recording state. Wrapping it in Option so the
        // borrow checker lets us move out of it inside the closure.
        let mut recorder_slot = Some(recorder);
        let mut next_state: Option<VoiceProfileState> = None;
        ui.horizontal(|ui| {
            if ui.add_enabled(can_save, save_btn).clicked() {
                if let Some(rec) = recorder_slot.take() {
                    let path = self.voice_profile_dir.join("user.wav");
                    match rec.stop_and_save(&path) {
                        Ok(saved) => {
                            let saved_str = saved.to_string_lossy().to_string();
                            self.send(TrayAction::SetVoiceProfile(Some(saved_str.clone())));
                            next_state = Some(VoiceProfileState::Idle {
                                saved_path: Some(saved_str),
                                last_error: None,
                            });
                        }
                        Err(e) => {
                            next_state = Some(VoiceProfileState::Idle {
                                saved_path: None,
                                last_error: Some(format!(
                                    "Falha ao salvar gravação: {}",
                                    e
                                )),
                            });
                        }
                    }
                }
            }

            ui.add_space(8.0);

            let cancel_btn = egui::Button::new(
                egui::RichText::new("✕  Cancelar")
                    .color(TEXT_PRIMARY)
                    .size(12.0),
            )
            .fill(BG_INPUT)
            .stroke(egui::Stroke::new(1.0, BORDER))
            .rounding(egui::Rounding::same(6.0));
            if ui.add(cancel_btn).clicked() {
                // Drop the recorder without saving.
                let _ = recorder_slot.take();
                next_state = Some(VoiceProfileState::Idle {
                    saved_path: None,
                    last_error: None,
                });
            }
        });

        // egui needs frequent repaints during recording so the timer
        // and level meter actually move.
        ui.ctx().request_repaint_after(std::time::Duration::from_millis(60));

        if let Some(state) = next_state {
            return state;
        }
        // Neither save nor cancel was clicked — keep recording.
        match recorder_slot {
            Some(recorder) => VoiceProfileState::Recording {
                recorder,
                started_at,
            },
            None => VoiceProfileState::Idle {
                saved_path: None,
                last_error: None,
            },
        }
    }

    /// Try to spin up a `VoiceRecorder` on the currently-selected mic
    /// device. Returns `Err(message)` when the device can't be opened
    /// — the caller surfaces this as the "last_error" line. We do NOT
    /// fall back to the OS default input here: that path silently
    /// recorded against the Hi-Fi Cable virtual device and produced
    /// 0-sample WAVs.
    fn start_recording(&self) -> Result<VoiceProfileState, String> {
        let device_name = self.mic_name();
        if device_name.trim().is_empty() || device_name == "(nenhum)" {
            return Err(
                "Selecione um microfone real no campo \"Dispositivo de entrada\" \
                 antes de gravar.".to_string(),
            );
        }
        let device = audio::device::find_input_device_by_name(&device_name)
            .map_err(|e| format!(
                "Não consegui abrir o microfone \"{}\": {}",
                device_name, e
            ))?;
        VoiceRecorder::start(&device)
            .map(|recorder| VoiceProfileState::Recording {
                recorder,
                started_at: std::time::Instant::now(),
            })
            .map_err(|e| format!("Falha ao iniciar a gravação: {}", e))
    }
}

fn metrics_table(ui: &mut egui::Ui, snapshot: &std::collections::HashMap<String, StageStats>) {
    if snapshot.is_empty() {
        ui.label(
            egui::RichText::new(
                "Sem amostras ainda — inicie a tradução para popular as métricas.",
            )
            .color(TEXT_SUBTLE)
            .size(11.5),
        );
        return;
    }

    egui::Grid::new("metrics_grid")
        .num_columns(4)
        .spacing(egui::vec2(16.0, 4.0))
        .striped(false)
        .show(ui, |ui| {
            ui.label(egui::RichText::new("Estágio").color(TEXT_MUTED).size(11.0));
            ui.label(egui::RichText::new("P50").color(TEXT_MUTED).size(11.0));
            ui.label(egui::RichText::new("P95").color(TEXT_MUTED).size(11.0));
            ui.label(egui::RichText::new("Amostras").color(TEXT_MUTED).size(11.0));
            ui.end_row();

            for (key, label) in METRIC_STAGES {
                let stats = match snapshot.get(*key) {
                    Some(s) => s,
                    None => continue,
                };
                let p50_ms = stats.p50.as_secs_f64() * 1000.0;
                let p95_ms = stats.p95.as_secs_f64() * 1000.0;
                ui.label(egui::RichText::new(*label).color(TEXT_PRIMARY).size(12.0));
                ui.label(egui::RichText::new(format_ms(p50_ms)).color(TEXT_PRIMARY).size(12.0));
                ui.label(egui::RichText::new(format_ms(p95_ms)).color(TEXT_PRIMARY).size(12.0));
                ui.label(
                    egui::RichText::new(format!("{}", stats.total_count))
                        .color(TEXT_SUBTLE)
                        .size(11.5),
                );
                ui.end_row();
            }
        });
}

fn format_ms(value_ms: f64) -> String {
    if value_ms < 10.0 {
        format!("{:.1} ms", value_ms)
    } else if value_ms < 1000.0 {
        format!("{:.0} ms", value_ms)
    } else {
        format!("{:.2} s", value_ms / 1000.0)
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
