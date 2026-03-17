use shared::PipelineCommand;
use tray_icon::menu::{Menu, MenuEvent, MenuId, MenuItem, PredefinedMenuItem, Submenu};
use tray_icon::{Icon, TrayIconBuilder};
use tracing;

pub struct TrayUi {
    toggle_id: MenuId,
    quit_id: MenuId,
    is_active: bool,
}

const ICON_SIZE: u32 = 32;

impl TrayUi {
    pub fn new() -> Result<Self, String> {
        let toggle_item = MenuItem::new("Start Translation", true, None);
        let quit_item = MenuItem::new("Quit", true, None);

        let toggle_id = toggle_item.id().clone();
        let quit_id = quit_item.id().clone();

        let menu = Menu::new();
        let direction_submenu = Submenu::new("Direction", true);
        let _ = direction_submenu.append(&MenuItem::new("EN → PT (Speaker)", true, None));
        let _ = direction_submenu.append(&MenuItem::new("PT → EN (Speaker)", true, None));

        let _ = menu.append(&direction_submenu);
        let _ = menu.append(&PredefinedMenuItem::separator());
        let _ = menu.append(&toggle_item);
        let _ = menu.append(&PredefinedMenuItem::separator());
        let _ = menu.append(&quit_item);

        let icon = create_default_icon();

        let _tray_icon = TrayIconBuilder::new()
            .with_menu(Box::new(menu))
            .with_tooltip("Meeting Translator - Inactive")
            .with_icon(icon)
            .build()
            .map_err(|e| format!("Failed to create tray icon: {}", e))?;

        // Keep tray_icon alive by leaking it — it will live for the entire process
        std::mem::forget(_tray_icon);

        tracing::info!("System tray UI created");

        Ok(Self {
            toggle_id,
            quit_id,
            is_active: false,
        })
    }

    pub fn process_events(&mut self) -> Option<TrayAction> {
        if let Ok(event) = MenuEvent::receiver().try_recv() {
            if event.id() == &self.toggle_id {
                self.is_active = !self.is_active;
                let action = if self.is_active {
                    tracing::info!("User toggled: ACTIVE");
                    TrayAction::Command(PipelineCommand::Start)
                } else {
                    tracing::info!("User toggled: INACTIVE");
                    TrayAction::Command(PipelineCommand::Stop)
                };
                return Some(action);
            }

            if event.id() == &self.quit_id {
                tracing::info!("User requested quit");
                return Some(TrayAction::Quit);
            }
        }
        None
    }

    pub fn is_active(&self) -> bool {
        self.is_active
    }
}

#[derive(Debug)]
pub enum TrayAction {
    Command(PipelineCommand),
    Quit,
}

fn create_default_icon() -> Icon {
    let mut rgba = Vec::with_capacity((ICON_SIZE * ICON_SIZE * 4) as usize);
    for _ in 0..ICON_SIZE * ICON_SIZE {
        rgba.push(60);  // R
        rgba.push(120); // G
        rgba.push(200); // B
        rgba.push(255); // A
    }
    Icon::from_rgba(rgba, ICON_SIZE, ICON_SIZE).expect("Failed to create icon")
}
