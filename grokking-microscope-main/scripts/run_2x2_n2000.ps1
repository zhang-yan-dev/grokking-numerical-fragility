# 2x2 structure-vs-cardinality re-run at n=2000 (n=3000 ceiling'd; need dynamic range).
$ErrorActionPreference = "Stop"
$venv = "F:\work\MAS\.venv\Scripts\python.exe"
$root = "F:\work\MAS"
$logdir = "$root\results\logs"
$env:PYTHONUTF8 = "1"; $env:OMP_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"

$jobs = @()
foreach ($fam in "muladd_m8", "addmul_m8", "muladd_m10", "addmul_m10") { foreach ($s in 0..9) {
  $jobs += @{ args = @("scripts\18_generality.py", $fam, "mono", "$s", "2000"); name = "18_${fam}_s${s}_n2000" }
} }

$CAP = 14
$queue = [System.Collections.ArrayList]@($jobs)
$running = New-Object System.Collections.ArrayList
$total = $queue.Count; $done = 0
Write-Output "LAUNCH 2x2-n2000 total=$total CAP=$CAP"
while ($queue.Count -gt 0 -or $running.Count -gt 0) {
  $still = New-Object System.Collections.ArrayList
  foreach ($r in $running) { if ($r.HasExited) { $done++ } else { [void]$still.Add($r) } }
  $running = $still
  while ($running.Count -lt $CAP -and $queue.Count -gt 0) {
    $j = $queue[0]; $queue.RemoveAt(0)
    $p = Start-Process -FilePath $venv -ArgumentList $j.args -WorkingDirectory $root -NoNewWindow -PassThru `
          -RedirectStandardOutput "$logdir\$($j.name).out" -RedirectStandardError "$logdir\$($j.name).err"
    [void]$running.Add($p)
  }
  Write-Output ("HEARTBEAT done={0}/{1} running={2} queued={3}" -f $done, $total, $running.Count, $queue.Count)
  Start-Sleep -Seconds 5
}
Write-Output "ALL DONE done=$done/$total"
