# PowerShell script to automate testing TheBrainV13.exe
$ErrorActionPreference = "Stop"

# Define Win32 API functions via Add-Type
$signature = @"
[DllImport("user32.dll", SetLastError = true)]
public static extern IntPtr FindWindowA(string lpClassName, string lpWindowName);

[DllImport("user32.dll", SetLastError = true)]
public static extern IntPtr FindWindowExA(IntPtr hwndParent, IntPtr hwndChildAfter, string lpszClass, string lpszWindow);

[DllImport("user32.dll", CharSet = CharSet.Ansi)]
public static extern IntPtr SendMessageA(IntPtr hWnd, uint Msg, IntPtr wParam, string lParam);

[DllImport("user32.dll", CharSet = CharSet.Ansi)]
public static extern IntPtr SendMessageA(IntPtr hWnd, uint Msg, IntPtr wParam, IntPtr lParam);

[DllImport("user32.dll")]
public static extern bool PostMessageA(IntPtr hWnd, uint Msg, IntPtr wParam, IntPtr lParam);

[DllImport("user32.dll")]
public static extern bool GetWindowTextA(IntPtr hWnd, System.Text.StringBuilder lpString, int nMaxCount);

[DllImport("user32.dll")]
public static extern bool EnumWindows(EnumWindowsProc lpEnumFunc, IntPtr lParam);
public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);
"@

Add-Type -MemberDefinition $signature -Name "Win32Utils" -Namespace "Win32"

Write-Output "Writing brain.conf directly..."
$conf = @"
# TheBrain v13.0 config - CORRECTED for CPU inference
lr=0.001000
dropout=0.2000
epochs=400
iso_thresh_min=0.5000
iso_thresh_max=0.9500
nu=0.0500
conf_thresh=0.8500
max_checkpoints=10
async_ops=1
log_level=1
watchdog_ms=600000
t_lr_max=0.001000
t_lr_min=0.000100
t_warmup=500
t_total=20000
t_wd=0.010000
t_grad_clip=1.0000
t_batch=4
t_ctx=256
use_swiglu=0
use_rmsnorm=0
tie_embeddings=0
temperature=0.8000
top_k=10
cot_think=192
cot_answer=512
early_stop=5
conv_max_tokens=64
conv_use_facts=1
conv_stream=1
conv_history_turns=8
guard_enabled=0
train_use_conv=1
train_use_text=1
vocab_file=vocab.bpak
model_file=model_v13.bin
embeds_file=embeds.bin
corpus_file=corpus.bin
sysinfo_tier=0
"@
[System.IO.File]::WriteAllText("brain.conf", $conf)

Write-Output "Deleting old log files..."
Remove-Item -Path "brain.log" -ErrorAction SilentlyContinue
Remove-Item -Path "brain_*.log" -ErrorAction SilentlyContinue

Write-Output "Launching TheBrainV13.exe..."
$proc = Start-Process -FilePath ".\TheBrainV13.exe" -PassThru

# Wait for the main window to be created
Write-Output "Waiting for window creation..."
Start-Sleep -Seconds 4

$hwnd = [IntPtr]::Zero

# Enum windows to find the main window
$enumProc = [Win32.Win32Utils+EnumWindowsProc] {
    param($h, $lpm)
    $sb = New-Object System.Text.StringBuilder 256
    [Win32.Win32Utils]::GetWindowTextA($h, $sb, 256) | Out-Null
    $title = $sb.ToString()
    if ($title -like "TheBrain*") {
        $script:hwnd = $h
        return $false # Stop enumerating
    }
    return $true # Continue
}

[Win32.Win32Utils]::EnumWindows($enumProc, [IntPtr]::Zero) | Out-Null

if ($hwnd -eq [IntPtr]::Zero) {
    # Fallback to MainWindowHandle
    $proc.Refresh()
    $hwnd = $proc.MainWindowHandle
}

if ($hwnd -eq [IntPtr]::Zero) {
    Write-Output "ERROR: Could not find main window."
    Stop-Process -Id $proc.Id -Force
    exit 1
}
Write-Output "Main window handle: $hwnd"

# Find the Input EDIT control (class name "EDIT")
$hwndEdit = [Win32.Win32Utils]::FindWindowExA($hwnd, [IntPtr]::Zero, "EDIT", $null)
if ($hwndEdit -eq [IntPtr]::Zero) {
    Write-Output "ERROR: Could not find input EDIT window."
    Stop-Process -Id $proc.Id -Force
    exit 1
}
Write-Output "EDIT control handle: $hwndEdit"

# Helper function to send a command and press Enter
function Send-Command {
    param([string]$cmd)
    Write-Output "Sending command: $cmd"
    # WM_SETTEXT = 0x000C
    [Win32.Win32Utils]::SendMessageA($hwndEdit, 0x000C, [IntPtr]::Zero, $cmd) | Out-Null
    Start-Sleep -Milliseconds 200
    # Send WM_COMMAND (0x0111) with BTN_SEND (1003) to the main window
    [Win32.Win32Utils]::SendMessageA($hwnd, 0x0111, [IntPtr]1003, $hwndEdit) | Out-Null
}

# Run the steps from COMMANDS.bat with proper waits
Send-Command "bpetrain data\conv\malware_qa.conv"
Write-Output "Waiting for tokenizer training..."
Start-Sleep -Seconds 12

Send-Command "pretrain"
Start-Sleep -Seconds 4

# Train on a few files to label them
Send-Command "train data\safe\math_utils.c 0"
Start-Sleep -Seconds 4
Send-Command "train data\dangerous\malware_patterns_injection.c 1"
Write-Output "Waiting for training task to finish..."
Start-Sleep -Seconds 15

# Now trigger the fulltrain command
Send-Command "fulltrain data 1"
Write-Output "Triggered fulltrain. Waiting for execution..."

# Monitor the process and the log
$timeout = 60 # 60 seconds
$elapsed = 0
while ($elapsed -lt $timeout) {
    if ($proc.HasExited) {
        Write-Output "Process exited! Exit code: $($proc.ExitCode)"
        break
    }
    # Read the log to check if it crashed
    if (Test-Path "brain.log") {
        $log = Get-Content "brain.log" -Tail 5
        if ($log -match "SEH:") {
            Write-Output "CRASH DETECTED IN LOG:"
            $log | Write-Output
            break
        }
    }
    Start-Sleep -Seconds 2
    $elapsed += 2
}

if (!$proc.HasExited) {
    Write-Output "Stopping process..."
    Stop-Process -Id $proc.Id -Force
}

# Read latest log contents to verify
Write-Output "Check latest log contents:"
if (Test-Path "brain.log") {
    Get-Content "brain.log" | Write-Output
}
if (Test-Path "brain_*.log") {
    $daily = Get-ChildItem "brain_*.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    Write-Output "Check daily log: $($daily.Name)"
    Get-Content $daily.FullName -Tail 30 | Write-Output
}
