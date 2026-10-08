param(
    [int]$Parallel = 4,
    [int]$Threads = 4
)

# Fill in the remainder of the wd grid: wd=0.0 (skip s7, already OURS) and wd=1.0 (all 10).
# Pure ASCII on purpose: PowerShell 5.1 mis-decodes non-ASCII in BOM-less .ps1 files.

$ErrorActionPreference = "Stop"
$root = "D:\research\run-ours"
$py = "D:\venvs\grokking\Scripts\python.exe"

Set-Location $root
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$logdir = Join-Path $root "logs"
New-Item -ItemType Directory -Force -Path $logdir | Out-Null

# jobs as "wd,seed" pairs. wd=0.0 seed 7 already done on this machine -> skip.
$jobs = @()
foreach ($s in 0..9) { if ($s -ne 7) { $jobs += ,@("0.0", $s) } }
foreach ($s in 0..9) { $jobs += ,@("1.0", $s) }

$total = $jobs.Count
Write-Output "launching $total runs: wd=0.0 (9, skip s7) + wd=1.0 (10), threads=$Threads, $Parallel concurrent"

$batch = @()
$done = 0

foreach ($j in $jobs) {
    $wd = $j[0]; $s = $j[1]
    $tag = $wd -replace '\.', 'p'
    $out = Join-Path $logdir "wd${tag}_s$s.log"
    $err = Join-Path $logdir "wd${tag}_s$s.err"
    $job = Start-Process -FilePath $py `
        -ArgumentList @("scripts\22_numerical_fragility.py", "$wd", "$Threads", "$s") `
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
