use std::path::PathBuf;
use std::process::Command;

/// Find a working Python 3.10+ interpreter.
///
/// Search order:
/// 1. PATH-based commands: "python", "python3", "py"
/// 2. Common Windows install locations (winget, python.org, system-wide)
///
/// Returns the full path or command name that works.
pub fn find_python() -> String {
    // 1. Try PATH-based commands first
    let path_candidates = ["python", "python3", "py"];
    for cmd in path_candidates {
        if is_valid_python(cmd) {
            return cmd.to_string();
        }
    }

    // 2. Search common Windows install locations
    let install_dirs = common_python_dirs();
    for dir in &install_dirs {
        let exe = dir.join("python.exe");
        if exe.exists() {
            let exe_str = exe.to_string_lossy().to_string();
            if is_valid_python(&exe_str) {
                return exe_str;
            }
        }
    }

    tracing::warn!("Python 3.10+ not found in PATH or common locations. Translation and TTS will fail.");
    "python".to_string()
}

fn is_valid_python(cmd: &str) -> bool {
    let result = Command::new(cmd)
        .arg("--version")
        .output();

    if let Ok(output) = result {
        if output.status.success() {
            let version = String::from_utf8_lossy(&output.stdout);
            return version.contains("Python 3.");
        }
    }
    false
}

/// Returns common Python install directories on Windows.
fn common_python_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();

    // winget / python.org per-user installs
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        for minor in (10..=15).rev() {
            dirs.push(PathBuf::from(format!(r"{}\Programs\Python\Python3{}", local, minor)));
        }
    }

    // python.org system-wide installs
    for minor in (10..=15).rev() {
        dirs.push(PathBuf::from(format!(r"C:\Python3{}", minor)));
        dirs.push(PathBuf::from(format!(r"C:\Program Files\Python3{}", minor)));
    }

    // Microsoft Store Python
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        dirs.push(PathBuf::from(format!(r"{}\Microsoft\WindowsApps", local)));
    }

    dirs
}
