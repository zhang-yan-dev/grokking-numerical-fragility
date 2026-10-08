param(
    [int]$Start = 0,
    [int]$End = 299,
    [int]$Parallel = 4,
    [string]$WD = "0.01",
    [int]$Threads = 4
)

# Batch launcher for script 22. Pure ASCII on purpose:
# PowerShell 5.1 mis-decodes non-ASCII in BOM-less .ps1 files.

$ErrorActionPreference = "Stop"
$root = "D:\research\run-ours"
$py = "D:\venvs\grokking\Scripts\python.exe"

Set-Location $root
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$logdir = Join-Path $root "logs"
New-Item -ItemType Directory -Force -Path $logdir | Out-Null

$total = ($End - $Start) + 1
Write-Output "launching $total runs: wd=$WD threads=$Threads seeds $Start..$End, $Parallel concurrent"
Write-Output "each run uses $Threads cpu threads -> $($Parallel * $Threads) threads total"

$batch = @()
$done = 0

foreach ($s in $Start..$End) {
    $out = Join-Path $logdir "s$s.log"
    $err = Join-Path $logdir "s$s.err"
    $job = Start-Process -FilePath $py `
        -ArgumentList @("scripts\22_numerical_fragility.py", "$WD", "$Threads", "$s") `
        -WorkingDirectory $root -NoNewWindow -PassThru `
        -RedirectStandardOutput $out -RedirectStandardError $err
    $batch += $job
    if ($batch.Count -ge $Parallel) {
        $batch | Wait-Process
        $done += $batch.Count
        Write-Output ("progress: $done / $total")
        $batch = @()
    }
}

if ($batch.Count -gt 0) {
    $batch | Wait-Process
    $done += $batch.Count
}
Write-Output "ALL_DONE $done / $total"
