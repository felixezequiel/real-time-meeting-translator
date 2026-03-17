use cpal::traits::{DeviceTrait, HostTrait};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DeviceError {
    #[error("No audio devices found")]
    NoDevicesFound,

    #[error("Device not found: {0}")]
    DeviceNotFound(String),

    #[error("Audio host error: {0}")]
    HostError(String),
}

#[derive(Debug, Clone)]
pub struct AudioDeviceInfo {
    pub name: String,
    pub is_input: bool,
    pub is_output: bool,
}

pub fn list_output_devices() -> Result<Vec<AudioDeviceInfo>, DeviceError> {
    let host = cpal::default_host();
    let devices = host
        .output_devices()
        .map_err(|e| DeviceError::HostError(e.to_string()))?;

    let mut result = Vec::new();
    for device in devices {
        let name = device
            .name()
            .unwrap_or_else(|_| "Unknown Device".to_string());
        result.push(AudioDeviceInfo {
            name,
            is_input: false,
            is_output: true,
        });
    }
    Ok(result)
}

pub fn list_input_devices() -> Result<Vec<AudioDeviceInfo>, DeviceError> {
    let host = cpal::default_host();
    let devices = host
        .input_devices()
        .map_err(|e| DeviceError::HostError(e.to_string()))?;

    let mut result = Vec::new();
    for device in devices {
        let name = device
            .name()
            .unwrap_or_else(|_| "Unknown Device".to_string());
        result.push(AudioDeviceInfo {
            name,
            is_input: true,
            is_output: false,
        });
    }
    Ok(result)
}

pub fn find_output_device_by_name(name_substring: &str) -> Result<cpal::Device, DeviceError> {
    let host = cpal::default_host();
    let devices = host
        .output_devices()
        .map_err(|e| DeviceError::HostError(e.to_string()))?;

    for device in devices {
        let device_name = device
            .name()
            .unwrap_or_else(|_| "Unknown".to_string());
        if device_name.to_lowercase().contains(&name_substring.to_lowercase()) {
            return Ok(device);
        }
    }
    Err(DeviceError::DeviceNotFound(name_substring.to_string()))
}

pub fn get_default_output_device() -> Result<cpal::Device, DeviceError> {
    let host = cpal::default_host();
    host.default_output_device()
        .ok_or(DeviceError::NoDevicesFound)
}

pub fn find_input_device_by_name(name_substring: &str) -> Result<cpal::Device, DeviceError> {
    let host = cpal::default_host();
    let devices = host
        .input_devices()
        .map_err(|e| DeviceError::HostError(e.to_string()))?;

    for device in devices {
        let device_name = device
            .name()
            .unwrap_or_else(|_| "Unknown".to_string());
        if device_name.to_lowercase().contains(&name_substring.to_lowercase()) {
            return Ok(device);
        }
    }
    Err(DeviceError::DeviceNotFound(name_substring.to_string()))
}

pub fn get_default_input_device() -> Result<cpal::Device, DeviceError> {
    let host = cpal::default_host();
    host.default_input_device()
        .ok_or(DeviceError::NoDevicesFound)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_output_devices_returns_without_error() {
        // May return empty list on CI, but should not error
        let result = list_output_devices();
        assert!(result.is_ok());
    }

    #[test]
    fn list_input_devices_returns_without_error() {
        let result = list_input_devices();
        assert!(result.is_ok());
    }

    #[test]
    fn find_nonexistent_device_returns_not_found() {
        let result = find_output_device_by_name("DEVICE_THAT_DOES_NOT_EXIST_12345");
        assert!(result.is_err());
    }
}
