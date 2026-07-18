# PowerShell script to automate training and converse verification
$ErrorActionPreference = "Stop"

# Define Win32 API functions
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
public static extern bool GetWindowTextA(IntPtr hWnd, System.Text.StringBuilder lpString, int nMaxCount);

[DllImport("user32.dll")]
public static extern bool EnumWindows(EnumWindowsProc lpEnumFunc, IntPtr lParam);
public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);
"@

if (-not ([Ref].Assembly.GetType("Win32.Win32Utils"))) {
    Add-Type -MemberDefinition $signature -Name "Win32Utils" -Namespace "Win32"
}

Write-Output "Deleting old log files..."
Remove-Item -Path "brain.log" -ErrorAction SilentlyContinue
Remove-Item -Path "brain_*.log" -ErrorAction SilentlyContinue

Write-Output "Launching TheBrainV13.exe..."
$proc = Start-Process -FilePath ".\TheBrainV13.exe" -PassThru

Write-Output "Waiting for window creation..."
Start-Sleep -Seconds 4

$hwnd = [IntPtr]::Zero
$enumProc = [Win32.Win32Utils+EnumWindowsProc] {
    param($h, $lpm)
    $sb = New-Object System.Text.StringBuilder 256
    [Win32.Win32Utils]::GetWindowTextA($h, $sb, 256) | Out-Null
    $title = $sb.ToString()
    if ($title -like "TheBrain*") {
        $script:hwnd = $h
        return $false
    }
    return $true
}

[Win32.Win32Utils]::EnumWindows($enumProc, [IntPtr]::Zero) | Out-Null

if ($hwnd -eq [IntPtr]::Zero) {
    $proc.Refresh()
    $hwnd = $proc.MainWindowHandle
}

if ($hwnd -eq [IntPtr]::Zero) {
    Write-Output "ERROR: Could not find main window."
    Stop-Process -Id $proc.Id -Force
    exit 1
}

$hwndEdit = [Win32.Win32Utils]::FindWindowExA($hwnd, [IntPtr]::Zero, "EDIT", $null)
if ($hwndEdit -eq [IntPtr]::Zero) {
    Write-Output "ERROR: Could not find input EDIT window."
    Stop-Process -Id $proc.Id -Force
    exit 1
}

function Send-Command {
    param([string]$cmd)
    Write-Output "Sending command: $cmd"
    [Win32.Win32Utils]::SendMessageA($hwndEdit, 0x000C, [IntPtr]::Zero, $cmd) | Out-Null
    Start-Sleep -Milliseconds 200
    [Win32.Win32Utils]::SendMessageA($hwnd, 0x0111, [IntPtr]1003, $hwndEdit) | Out-Null
}

# 1. Train tokenizer
Send-Command "bpetrain data\conv\malware_qa.conv"
Start-Sleep -Seconds 12

# 2. Init transformer
Send-Command "pretrain"
Start-Sleep -Seconds 4

# 3. Label safe/dangerous files
Send-Command "train data\safe\string_utils.c 0"
Start-Sleep -Seconds 2
Send-Command "train data\safe\math_utils.c 0"
Start-Sleep -Seconds 2
Send-Command "train data\dangerous\malware_patterns_injection.c 1"
Start-Sleep -Seconds 2
Send-Command "train data\dangerous\malware_patterns_persistence.c 1"
Start-Sleep -Seconds 12

# 4. Trigger fulltrain for 50 epochs
Send-Command "fulltrain data 50"
Write-Output "Triggered fulltrain 50. Waiting for completion..."

$timeout = 2400 # 40 minutes max
$elapsed = 0
$completed = $false

while ($elapsed -lt $timeout) {
    if ($proc.HasExited) {
        Write-Output "Process exited prematurely!"
        break
    }
    
    # Check brain.log for completion
    if (Test-Path "brain.log") {
        $logContent = Get-Content "brain.log"
        if ($logContent -match "train_loop_mixed done") {
            Write-Output "Training completed successfully in log!"
            $completed = $true
            break
        }
    }
    
    Start-Sleep -Seconds 5
    $elapsed += 5
}

if ($completed) {
    Start-Sleep -Seconds 2
    Send-Command "How does ransomware work?"
    Write-Output "Sent question. Waiting for response generation..."
    Start-Sleep -Seconds 8
} else {
    Write-Output "ERROR: Training did not complete in time."
}


# Read daily log contents to verify output
$dailyLog = Get-ChildItem "brain_*.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($dailyLog) {
    Write-Output "---------------- DAILY LOG PATH: $($dailyLog.FullName) ----------------"
    Get-Content $dailyLog.FullName -Tail 50 | Write-Output
}
