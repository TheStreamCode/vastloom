# Purge the Windows standby list so the next model run reads from the device instead of the file cache.
#
# This is a benchmark helper, not part of normal execution: it flushes the system file cache and can
# make unrelated applications slower until their pages are read again. Run it only for a controlled
# cold-cache measurement, in an elevated PowerShell, and do not leave it in a loop.
#
# Usage:
#   Start-Process pwsh -Verb RunAs -ArgumentList '-NoProfile -File scripts/purge-standby.ps1 -Force'
#   # or from an already elevated shell:
#   pwsh -NoProfile -File scripts/purge-standby.ps1 -Force
#
# After it reports success, run the measurement immediately:
#   build\release\vastloom.exe deepseek-chain-probe ... --json
# and repeat the same command with --expert-pack / --io-cache bypass for the A/B.
[CmdletBinding()]
param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsPrincipal]::new($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Error "This script needs an elevated PowerShell (Run as Administrator); the standby-list purge requires SeProfileSingleProcessPrivilege."
    exit 1
}

if (-not $Force) {
    $answer = Read-Host "This flushes the Windows standby list for the whole machine. Continue? [y/N]"
    if ($answer -notmatch '^[yY]$') {
        Write-Host "Cancelled."
        exit 0
    }
}

$source = @'
using System;
using System.Runtime.InteropServices;

public static class VastloomStandbyPurge
{
    private const int SystemMemoryListInformation = 80;
    private const int MemoryPurgeStandbyList = 4;
    private const uint TokenAdjustPrivileges = 0x0020;
    private const uint TokenQuery = 0x0008;
    private const uint SePrivilegeEnabled = 0x0002;

    [StructLayout(LayoutKind.Sequential)]
    private struct Luid { public uint LowPart; public int HighPart; }

    [StructLayout(LayoutKind.Sequential)]
    private struct TokenPrivileges
    {
        public uint PrivilegeCount;
        public Luid Luid;
        public uint Attributes;
    }

    [DllImport("ntdll.dll")]
    private static extern int NtSetSystemInformation(int SystemInformationClass,
                                                     IntPtr SystemInformation,
                                                     int SystemInformationLength);

    [DllImport("advapi32.dll", SetLastError = true)]
    private static extern bool OpenProcessToken(IntPtr ProcessHandle, uint DesiredAccess,
                                                 out IntPtr TokenHandle);

    [DllImport("advapi32.dll", SetLastError = true)]
    private static extern bool LookupPrivilegeValue(string SystemName, string Name, out Luid Luid);

    [DllImport("advapi32.dll", SetLastError = true)]
    private static extern bool AdjustTokenPrivileges(IntPtr TokenHandle, bool DisableAllPrivileges,
                                                      ref TokenPrivileges NewState,
                                                      int BufferLength, IntPtr PreviousState,
                                                      IntPtr ReturnLength);

    [DllImport("kernel32.dll")]
    private static extern IntPtr GetCurrentProcess();

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool CloseHandle(IntPtr Handle);

    public static string Purge()
    {
        IntPtr token = IntPtr.Zero;
        if (!OpenProcessToken(GetCurrentProcess(), TokenAdjustPrivileges | TokenQuery, out token))
            return "OpenProcessToken failed: " + Marshal.GetLastWin32Error();
        try
        {
            Luid luid;
            if (!LookupPrivilegeValue(null, "SeProfileSingleProcessPrivilege", out luid))
                return "LookupPrivilegeValue failed: " + Marshal.GetLastWin32Error();
            var privileges = new TokenPrivileges
            {
                PrivilegeCount = 1,
                Luid = luid,
                Attributes = SePrivilegeEnabled
            };
            if (!AdjustTokenPrivileges(token, false, ref privileges, 0, IntPtr.Zero, IntPtr.Zero))
                return "AdjustTokenPrivileges failed: " + Marshal.GetLastWin32Error();
        }
        finally
        {
            CloseHandle(token);
        }

        IntPtr buffer = Marshal.AllocHGlobal(sizeof(int));
        try
        {
            Marshal.WriteInt32(buffer, MemoryPurgeStandbyList);
            int status = NtSetSystemInformation(SystemMemoryListInformation, buffer, sizeof(int));
            return status == 0 ? "ok" : "NtSetSystemInformation failed: " + status;
        }
        finally
        {
            Marshal.FreeHGlobal(buffer);
        }
    }
}
'@

Add-Type -TypeDefinition $source -Language CSharp

$result = [VastloomStandbyPurge]::Purge()
if ($result -eq "ok") {
    Write-Host "Standby list purge requested successfully. Run the cold-cache measurement now."
    exit 0
}
Write-Error "Standby list purge failed: $result"
exit 1
