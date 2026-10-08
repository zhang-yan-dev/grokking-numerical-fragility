Set-Location F:\work\MAS
$env:PYTHONUTF8 = 1
$env:PYTHONIOENCODING = "utf-8"
$jobs = @()
# (1) extend the t=1 vs t=4 reduction-order experiment to seeds 10-29 (t16 omitted: == t4)
foreach ($t in 1, 4) {
  foreach ($s in 10..29) {
    $args = @("scripts\22_numerical_fragility.py", "0.01", "$t", "$s")
    $jobs += Start-Process -FilePath "F:\work\MAS\.venv\Scripts\python.exe" -ArgumentList $args -WorkingDirectory "F:\work\MAS" -NoNewWindow -PassThru -RedirectStandardOutput "results\22_wd0.01_t${t}_s${s}.log" -RedirectStandardError "results\22_wd0.01_t${t}_s${s}.err"
  }
}
# (2) weight-decay positive control: wd {0.0,0.1,1.0} x seeds 0-9 at threads=4
foreach ($wd in "0.0", "0.1", "1.0") {
  foreach ($s in 0..9) {
    $args = @("scripts\22_numerical_fragility.py", "$wd", "4", "$s")
    $jobs += Start-Process -FilePath "F:\work\MAS\.venv\Scripts\python.exe" -ArgumentList $args -WorkingDirectory "F:\work\MAS" -NoNewWindow -PassThru -RedirectStandardOutput "results\22_wd${wd}_t4_s${s}.log" -RedirectStandardError "results\22_wd${wd}_t4_s${s}.err"
  }
}
Write-Output ("launched " + $jobs.Count + " phase-2 jobs (40 reduction-order extend + 30 wd control)")
$jobs | Wait-Process
Write-Output "PHASE2_DONE"
