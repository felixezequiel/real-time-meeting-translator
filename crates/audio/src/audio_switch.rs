/// Switches the Windows default audio output device using a PowerShell script.
/// Returns the previous default device name so it can be restored later.
use std::path::Path;
use std::process::Command;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AudioSwitchError {
    #[error("PowerShell script failed: {0}")]
    ScriptFailed(String),

    #[error("Script not found: {0}")]
    ScriptNotFound(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Set VB-Cable as the system default output device.
/// Returns the name of the previous default device (for later restore).
pub fn set_default_output_device(
    script_path: &Path,
    device_name: &str,
) -> Result<String, AudioSwitchError> {
    if !script_path.exists() {
        return Err(AudioSwitchError::ScriptNotFound(
            script_path.display().to_string(),
        ));
    }

    let output = Command::new("powershell")
        .args([
            "-ExecutionPolicy", "Bypass",
            "-File", &script_path.to_string_lossy(),
            "-Action", "set-default",
            "-DeviceName", device_name,
        ])
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(AudioSwitchError::ScriptFailed(stderr.to_string()));
    }

    let previous_device = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Ok(previous_device)
}

/// Get the current default output device name.
pub fn get_default_output_device(script_path: &Path) -> Result<String, AudioSwitchError> {
    if !script_path.exists() {
        return Err(AudioSwitchError::ScriptNotFound(
            script_path.display().to_string(),
        ));
    }

    let output = Command::new("powershell")
        .args([
            "-ExecutionPolicy", "Bypass",
            "-File", &script_path.to_string_lossy(),
            "-Action", "get-default",
        ])
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(AudioSwitchError::ScriptFailed(stderr.to_string()));
    }

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Set the system default input (microphone) device.
/// Returns the name of the previous default input device (for later restore).
pub fn set_default_input_device(
    script_path: &Path,
    device_name: &str,
) -> Result<String, AudioSwitchError> {
    if !script_path.exists() {
        return Err(AudioSwitchError::ScriptNotFound(
            script_path.display().to_string(),
        ));
    }

    let output = Command::new("powershell")
        .args([
            "-ExecutionPolicy", "Bypass",
            "-File", &script_path.to_string_lossy(),
            "-Action", "set-default-input",
            "-DeviceName", device_name,
        ])
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(AudioSwitchError::ScriptFailed(stderr.to_string()));
    }

    let previous_device = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Ok(previous_device)
}
