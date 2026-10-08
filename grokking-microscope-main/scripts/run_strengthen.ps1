# Concurrency-throttled launcher for the reviewer-strengthening grids.
# CPU cap 14, GPU cap 4 -> leaves headroom for the 4 unrelated other-project python processes.
$ErrorActionPreference = "Stop"
$venv = "F:\work\MAS\.venv\Scripts\python.exe"
$root = "F:\work\MAS"
$logdir = "$root\results\logs"
New-Item -ItemType Directory -Force $logdir | Out-Null
$env:PYTHONUTF8 = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"

$jobs = @()
# Exp3 FIRST (heaviest, n=3000 ~900s each): front-load to avoid a long single-job tail.
# structure 2x2: {muladd_m8,addmul_m8,muladd_m10,addmul_m10} x n3000 x seed0-9
foreach ($fam in "muladd_m8", "addmul_m8", "muladd_m10", "addmul_m10") { foreach ($s in 0..9) {
  $jobs += @{ args = @("scripts\18_generality.py", $fam, "mono", "$s", "3000"); gpu = $false; name = "18_${fam}_s$s" }
} }
# Exp1 device: seeds 10-29 x {cpu,cuda}
foreach ($s in 10..29) {
  $jobs += @{ args = @("scripts\26_device_control.py", "cpu",  "$s"); gpu = $false; name = "26_cpu_s$s" }
  $jobs += @{ args = @("scripts\26_device_control.py", "cuda", "$s"); gpu = $true;  name = "26_cuda_s$s" }
}
# Exp4 bitident train (compare done separately after)
$jobs += @{ args = @("scripts\34_thread_bitident.py", "train", "4",  "0"); gpu = $false; name = "34_t4_s0" }
$jobs += @{ args = @("scripts\34_thread_bitident.py", "train", "16", "0"); gpu = $false; name = "34_t16_s0" }
# Exp2 cardinality 3-input: M{5..10} x n{300,600} x seed0-9 (light; fills CPU gaps)
foreach ($M in 5..10) { foreach ($n in 300, 600) { foreach ($s in 0..9) {
  $jobs += @{ args = @("scripts\33_cardinality_3input.py", "$M", "$n", "$s"); gpu = $false; name = "33_m${M}_n${n}_s$s" }
} } }

$CPU_CAP = 14; $GPU_CAP = 4
$queue = [System.Collections.ArrayList]@($jobs)
$running = New-Object System.Collections.ArrayList
$total = $queue.Count; $done = 0
Write-Output "LAUNCH total=$total CPU_CAP=$CPU_CAP GPU_CAP=$GPU_CAP"

while ($queue.Count -gt 0 -or $running.Count -gt 0) {
  # reap
  $stillRun = New-Object System.Collections.ArrayList
  foreach ($r in $running) {
    if ($r.proc.HasExited) { $done++ } else { [void]$stillRun.Add($r) }
  }
  $running = $stillRun
  $cpuRun = @($running | Where-Object { -not $_.gpu }).Count
  $gpuRun = @($running | Where-Object { $_.gpu }).Count
  # launch eligible
  $progress = $true
  while ($progress) {
    $progress = $false
    for ($i = 0; $i -lt $queue.Count; $i++) {
      $j = $queue[$i]
      if ($j.gpu) { if ($gpuRun -ge $GPU_CAP) { continue } } else { if ($cpuRun -ge $CPU_CAP) { continue } }
      $p = Start-Process -FilePath $venv -ArgumentList $j.args -WorkingDirectory $root -NoNewWindow -PassThru `
            -RedirectStandardOutput "$logdir\$($j.name).out" -RedirectStandardError "$logdir\$($j.name).err"
      [void]$running.Add(@{ proc = $p; gpu = $j.gpu })
      if ($j.gpu) { $gpuRun++ } else { $cpuRun++ }
      $queue.RemoveAt($i)
      $progress = $true
      break
    }
  }
  Write-Output ("HEARTBEAT done={0}/{1} running={2} (cpu={3} gpu={4}) queued={5}" -f $done, $total, $running.Count, $cpuRun, $gpuRun, $queue.Count)
  Start-Sleep -Seconds 5
}
Write-Output "ALL DONE done=$done/$total"
