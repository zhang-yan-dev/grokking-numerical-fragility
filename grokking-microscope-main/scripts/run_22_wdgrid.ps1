Set-Location F:\work\MAS
$env:PYTHONUTF8 = 1
$env:PYTHONIOENCODING = "utf-8"
Write-Output "wd-grid: wd {0.0,0.1,1.0} x seeds 0-9 at threads=4 (wd=0.01,t=4 comes from thread grid)"
$jobs = @()
foreach ($wd in "0.0", "0.1", "1.0") {
  foreach ($s in 0..9) {
    $args = @("scripts\22_numerical_fragility.py", "$wd", "4", "$s")
    $jobs += Start-Process -FilePath "F:\work\MAS\.venv\Scripts\python.exe" -ArgumentList $args -WorkingDirectory "F:\work\MAS" -NoNewWindow -PassThru -RedirectStandardOutput "results\22_wd${wd}_t4_s${s}.log" -RedirectStandardError "results\22_wd${wd}_t4_s${s}.err"
  }
}
Write-Output ("launched " + $jobs.Count + " wd-grid jobs")
$jobs | Wait-Process
Write-Output "WD_GRID_DONE"
