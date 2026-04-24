//! Automatic audio ducking for other applications during TTS playback.
//!
//! Uses the Windows Audio Session API (WASAPI) to reduce the volume of all
//! audio sessions on the default render endpoint except our own while
//! translated speech is playing, then restore them when the TTS buffer
//! goes silent for longer than the debounce window.
//!
//! See `docs/adr/0002-audio-ducking-wasapi.md` for the decision rationale.

use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};
use thiserror::Error;

/// Target gain applied to other apps while TTS is speaking.
/// 0.25 keeps context audible (you still hear the original voice softly)
/// while making the translation the clear foreground.
pub const DEFAULT_DUCK_VOLUME: f32 = 0.25;

#[derive(Debug, Error)]
pub enum DuckerError {
    #[error("WASAPI initialization failed: {0}")]
    Init(String),

    #[error("Ducker worker thread panicked")]
    WorkerGone,
}

/// Commands sent from the pipeline supervisor to the ducker worker thread.
enum DuckCommand {
    /// Enumerate audio sessions and lower volume of every session except ours.
    Duck,
    /// Restore every session whose volume we changed back to its prior value.
    Restore,
    /// Stop the worker (also triggers a final Restore so we don't leak
    /// attenuated volumes to other apps after exit).
    Shutdown,
}

pub struct AudioDucker {
    command_tx: Sender<DuckCommand>,
    worker: Option<JoinHandle<()>>,
}

/// Clonable remote-control for a running `AudioDucker`. Holds only a sender
/// handle — dropping a handle does **not** stop the worker, only dropping
/// the owning `AudioDucker` does (which also triggers a final Restore).
#[derive(Clone)]
pub struct DuckerHandle {
    command_tx: Sender<DuckCommand>,
}

impl AudioDucker {
    pub fn new(target_volume: f32) -> Result<Self, DuckerError> {
        let (tx, rx) = mpsc::channel();
        let worker = thread::Builder::new()
            .name("audio-ducker".to_string())
            .spawn(move || worker_main(rx, target_volume))
            .map_err(|e| DuckerError::Init(format!("spawn failed: {}", e)))?;
        Ok(Self { command_tx: tx, worker: Some(worker) })
    }

    pub fn handle(&self) -> DuckerHandle {
        DuckerHandle { command_tx: self.command_tx.clone() }
    }

    pub fn duck(&self) {
        let _ = self.command_tx.send(DuckCommand::Duck);
    }

    pub fn restore(&self) {
        let _ = self.command_tx.send(DuckCommand::Restore);
    }
}

impl DuckerHandle {
    pub fn duck(&self) {
        let _ = self.command_tx.send(DuckCommand::Duck);
    }

    pub fn restore(&self) {
        let _ = self.command_tx.send(DuckCommand::Restore);
    }
}

impl Drop for AudioDucker {
    fn drop(&mut self) {
        let _ = self.command_tx.send(DuckCommand::Shutdown);
        if let Some(handle) = self.worker.take() {
            let _ = handle.join();
        }
    }
}

// ─── Platform-specific worker ────────────────────────────────────────────────

#[cfg(windows)]
fn worker_main(rx: Receiver<DuckCommand>, target_volume: f32) {
    windows_impl::run(rx, target_volume);
}

#[cfg(not(windows))]
fn worker_main(rx: Receiver<DuckCommand>, _target_volume: f32) {
    // Non-Windows platforms: ducker is a no-op. Drain commands until shutdown.
    while let Ok(cmd) = rx.recv() {
        if matches!(cmd, DuckCommand::Shutdown) {
            break;
        }
    }
}

#[cfg(windows)]
mod windows_impl {
    use super::{DuckCommand, Receiver};
    use std::collections::HashMap;
    use tracing;
    use windows::core::Interface;
    use windows::Win32::Media::Audio::{
        eMultimedia, eRender, IAudioSessionControl2, IAudioSessionEnumerator,
        IAudioSessionManager2, IMMDeviceEnumerator, ISimpleAudioVolume, MMDeviceEnumerator,
    };
    use windows::Win32::System::Com::{
        CoCreateInstance, CoInitializeEx, CoUninitialize, CLSCTX_ALL, COINIT_MULTITHREADED,
    };

    /// Volumes we lowered and must restore, keyed by process id.
    type SavedVolumes = HashMap<u32, (ISimpleAudioVolume, f32)>;

    pub fn run(rx: Receiver<DuckCommand>, target_volume: f32) {
        // Safety: CoInitializeEx returns S_OK or S_FALSE on success. Anything
        // else means COM can't work here — bail and leave the channel draining.
        let hr = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };
        if hr.is_err() {
            tracing::error!("CoInitializeEx failed: {:?} — ducking disabled", hr);
            // Still drain commands so the Sender doesn't block forever.
            while let Ok(cmd) = rx.recv() {
                if matches!(cmd, DuckCommand::Shutdown) {
                    break;
                }
            }
            return;
        }

        let own_pid = std::process::id();
        let mut saved: SavedVolumes = HashMap::new();

        while let Ok(cmd) = rx.recv() {
            match cmd {
                DuckCommand::Duck => {
                    if let Err(e) = apply_duck(&mut saved, own_pid, target_volume) {
                        tracing::warn!("duck failed: {}", e);
                    }
                }
                DuckCommand::Restore => {
                    restore_all(&mut saved);
                }
                DuckCommand::Shutdown => {
                    restore_all(&mut saved);
                    break;
                }
            }
        }

        unsafe { CoUninitialize() };
    }

    fn apply_duck(
        saved: &mut SavedVolumes,
        own_pid: u32,
        target_volume: f32,
    ) -> Result<(), String> {
        let sessions = enumerate_sessions().map_err(|e| format!("enumerate: {}", e))?;

        for (pid, volume_ctrl) in sessions {
            if pid == 0 || pid == own_pid {
                continue;
            }
            if saved.contains_key(&pid) {
                // Already ducked in a previous Duck without Restore in between.
                continue;
            }
            let original = unsafe { volume_ctrl.GetMasterVolume() }.unwrap_or(1.0);
            let guid = std::ptr::null();
            let set_result = unsafe { volume_ctrl.SetMasterVolume(target_volume, guid) };
            if set_result.is_ok() {
                saved.insert(pid, (volume_ctrl, original));
            }
        }
        Ok(())
    }

    fn restore_all(saved: &mut SavedVolumes) {
        for (_pid, (volume_ctrl, original)) in saved.drain() {
            let guid = std::ptr::null();
            let _ = unsafe { volume_ctrl.SetMasterVolume(original, guid) };
        }
    }

    /// Walks the default render endpoint's audio sessions and returns
    /// `(process_id, volume_control)` for each one. Any single failing
    /// session is skipped — we never want a rogue session to kill ducking
    /// for all the others.
    fn enumerate_sessions() -> windows::core::Result<Vec<(u32, ISimpleAudioVolume)>> {
        let enumerator: IMMDeviceEnumerator =
            unsafe { CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL) }?;
        let device = unsafe { enumerator.GetDefaultAudioEndpoint(eRender, eMultimedia) }?;

        let session_mgr: IAudioSessionManager2 =
            unsafe { device.Activate(CLSCTX_ALL, None) }?;
        let sessions: IAudioSessionEnumerator =
            unsafe { session_mgr.GetSessionEnumerator() }?;
        let count = unsafe { sessions.GetCount() }?;

        let mut result = Vec::with_capacity(count as usize);
        for i in 0..count {
            let session = match unsafe { sessions.GetSession(i) } {
                Ok(s) => s,
                Err(_) => continue,
            };
            let session2: IAudioSessionControl2 = match session.cast() {
                Ok(s) => s,
                Err(_) => continue,
            };
            let pid = match unsafe { session2.GetProcessId() } {
                Ok(p) => p,
                Err(_) => continue,
            };
            let volume_ctrl: ISimpleAudioVolume = match session.cast() {
                Ok(v) => v,
                Err(_) => continue,
            };
            result.push((pid, volume_ctrl));
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_duck_volume_is_audible_but_attenuated() {
        // Sanity guards — not functional tests (WASAPI can't be unit-tested here).
        assert!(DEFAULT_DUCK_VOLUME > 0.0);
        assert!(DEFAULT_DUCK_VOLUME < 1.0);
    }

    #[test]
    fn ducker_can_be_created_and_dropped() {
        // Spawns the worker thread; Drop must shut it down cleanly.
        let ducker = AudioDucker::new(DEFAULT_DUCK_VOLUME).expect("ducker failed to start");
        drop(ducker);
    }
}
