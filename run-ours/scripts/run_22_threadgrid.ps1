Set-Location F:\work\MAS
$env:PYTHONUTF8 = 1
$env:PYTHONIOENCODING = "utf-8"
& .\.venv\Scripts\python.exe -m py_compile scripts\22_numerical_fragility.py
if (-not $?) { Write-Output "SYNTAX_FAIL"; exit 1 }
Write-Output "syntax OK"
$jobs = @()
foreach ($t in 1, 4, 16) {
  foreach ($s in 0..9) {
    $args = @("scripts\22_numerical_fragility.py", "0.01", "$t", "$s")
    $jobs += Start-Process -FilePath "F:\work\MAS\.venv\Scripts\python.exe" -ArgumentList $args -WorkingDirectory "F:\work\MAS" -NoNewWindow -PassThru -RedirectStandardOutput "results\22_wd0.01_t${t}_s${s}.log" -RedirectStandardError "results\22_wd0.01_t${t}_s${s}.err"
  }
}
Write-Output ("launched " + $jobs.Count + " thread-grid jobs")
$jobs | Wait-Process
Write-Output "THREAD_GRID_DONE"
