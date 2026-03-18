param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("get-default", "set-default", "list")]
    [string]$Action,

    [string]$DeviceName
)

# Minimal C# COM interop for MMDeviceEnumerator + IPolicyConfig
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

// ── MMDevice API (documented) ──────────────────────────────────────────────

[Flags] public enum DEVICE_STATE : uint { ACTIVE = 0x1 }
public enum EDataFlow { eRender = 0, eCapture = 1, eAll = 2 }
public enum ERole { eConsole = 0, eMultimedia = 1, eCommunications = 2 }

[Guid("D666063F-1587-4E43-81F1-B948E807363F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
public interface IMMDevice {
    int Activate(ref Guid iid, int dwClsCtx, IntPtr pActivationParams, [MarshalAs(UnmanagedType.IUnknown)] out object ppInterface);
    int OpenPropertyStore(int stgmAccess, out IPropertyStore ppProperties);
    int GetId([MarshalAs(UnmanagedType.LPWStr)] out string ppstrId);
    int GetState(out uint pdwState);
}

[Guid("0BD7A1BE-7A1A-44DB-8397-CC5392387B5E"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
public interface IMMDeviceCollection {
    int GetCount(out int pcDevices);
    int Item(int nDevice, out IMMDevice ppDevice);
}

[Guid("A95664D2-9614-4F35-A746-DE8DB63617E6"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
public interface IMMDeviceEnumerator {
    int EnumAudioEndpoints(EDataFlow dataFlow, uint dwStateMask, out IMMDeviceCollection ppDevices);
    int GetDefaultAudioEndpoint(EDataFlow dataFlow, ERole role, out IMMDevice ppEndpoint);
    // remaining methods omitted
}

[ComImport, Guid("BCDE0395-E52F-467C-8E3D-C4579291692E")]
public class MMDeviceEnumerator {}

[Guid("886d8eeb-8cf2-4446-8d02-cdba1dbdcf99"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
public interface IPropertyStore {
    int GetCount(out int cProps);
    int GetAt(int iProp, out PROPERTYKEY pkey);
    int GetValue(ref PROPERTYKEY key, out PROPVARIANT pv);
}

[StructLayout(LayoutKind.Sequential)]
public struct PROPERTYKEY {
    public Guid fmtid;
    public int pid;
}

[StructLayout(LayoutKind.Sequential)]
public struct PROPVARIANT {
    public ushort vt;
    public ushort wReserved1, wReserved2, wReserved3;
    public IntPtr val1;
    public IntPtr val2;
}

// ── IPolicyConfig (undocumented but stable since Vista) ────────────────────

[Guid("F8679F50-850A-41CF-9C72-430F290290C8"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
public interface IPolicyConfig {
    int GetMixFormat(string pwstrDeviceId, IntPtr ppFormat);
    int GetDeviceFormat(string pwstrDeviceId, int bDefault, IntPtr ppFormat);
    int ResetDeviceFormat(string pwstrDeviceId);
    int SetDeviceFormat(string pwstrDeviceId, IntPtr pEndpointFormat, IntPtr mixFormat);
    int GetProcessingPeriod(string pwstrDeviceId, int bDefault, IntPtr pmftDefaultPeriod, IntPtr pmftMinimumPeriod);
    int SetProcessingPeriod(string pwstrDeviceId, IntPtr pmftPeriod);
    int GetShareMode(string pwstrDeviceId, IntPtr pMode);
    int SetShareMode(string pwstrDeviceId, IntPtr mode);
    int GetPropertyValue(string pwstrDeviceId, IntPtr key, IntPtr pv);
    int SetPropertyValue(string pwstrDeviceId, IntPtr key, IntPtr pv);
    int SetDefaultEndpoint(string pwstrDeviceId, int eRole);
    int SetEndpointVisibility(string pwstrDeviceId, int bVisible);
}

[ComImport, Guid("870AF99C-171D-4F9E-AF0D-E63DF40C2BC9")]
public class PolicyConfigClient {}

// ── Helper class ───────────────────────────────────────────────────────────

public static class AudioSwitcher {
    static readonly PROPERTYKEY PKEY_Device_FriendlyName = new PROPERTYKEY {
        fmtid = new Guid("a45c254e-df1c-4efd-8020-67d146a850e0"),
        pid = 14
    };

    public static string GetDeviceName(IMMDevice device) {
        IPropertyStore store;
        device.OpenPropertyStore(0, out store);
        var key = PKEY_Device_FriendlyName;
        PROPVARIANT pv;
        store.GetValue(ref key, out pv);
        return Marshal.PtrToStringUni(pv.val1) ?? "Unknown";
    }

    public static string GetDeviceId(IMMDevice device) {
        string id;
        device.GetId(out id);
        return id;
    }

    public static string GetDefaultOutputName() {
        var enumerator = (IMMDeviceEnumerator)new MMDeviceEnumerator();
        IMMDevice device;
        enumerator.GetDefaultAudioEndpoint(EDataFlow.eRender, ERole.eConsole, out device);
        return GetDeviceName(device);
    }

    public static void SetDefaultByName(string targetName) {
        var enumerator = (IMMDeviceEnumerator)new MMDeviceEnumerator();
        IMMDeviceCollection collection;
        enumerator.EnumAudioEndpoints(EDataFlow.eRender, (uint)DEVICE_STATE.ACTIVE, out collection);
        int count;
        collection.GetCount(out count);

        string targetLower = targetName.ToLowerInvariant();
        for (int i = 0; i < count; i++) {
            IMMDevice device;
            collection.Item(i, out device);
            string name = GetDeviceName(device);
            if (name.ToLowerInvariant().Contains(targetLower)) {
                string id = GetDeviceId(device);
                var config = (IPolicyConfig)new PolicyConfigClient();
                config.SetDefaultEndpoint(id, (int)ERole.eConsole);
                config.SetDefaultEndpoint(id, (int)ERole.eMultimedia);
                config.SetDefaultEndpoint(id, (int)ERole.eCommunications);
                return;
            }
        }
        throw new Exception("Device not found: " + targetName);
    }

    public static string[] ListOutputDevices() {
        var enumerator = (IMMDeviceEnumerator)new MMDeviceEnumerator();
        IMMDeviceCollection collection;
        enumerator.EnumAudioEndpoints(EDataFlow.eRender, (uint)DEVICE_STATE.ACTIVE, out collection);
        int count;
        collection.GetCount(out count);
        var names = new string[count];
        for (int i = 0; i < count; i++) {
            IMMDevice device;
            collection.Item(i, out device);
            names[i] = GetDeviceName(device);
        }
        return names;
    }
}
"@ -ErrorAction Stop

switch ($Action) {
    "get-default" {
        [AudioSwitcher]::GetDefaultOutputName()
    }
    "set-default" {
        if (-not $DeviceName) {
            Write-Error "DeviceName required for set-default"
            exit 1
        }
        $previous = [AudioSwitcher]::GetDefaultOutputName()
        [AudioSwitcher]::SetDefaultByName($DeviceName)
        # Return previous device name so caller can restore it
        $previous
    }
    "list" {
        [AudioSwitcher]::ListOutputDevices()
    }
}
