Set-Location F:\work\MAS
$env:PYTHONUTF8 = 1
$env:PYTHONIOENCODING = "utf-8"
& .\.venv\Scripts\python.exe -m py_compile scripts\26_device_control.py
if (-not $?) { Write-Output "SYNTAX_FAIL"; exit 1 }
Write-Output "syntax OK"
$jobs = @()
foreach ($dev in "cpu", "cuda") {
  foreach ($s in 0..9) {
    $args = @("scripts\26_device_control.py", "$dev", "$s")
    $jobs += Start-Process -FilePath "F:\work\MAS\.venv\Scripts\python.exe" -ArgumentList $args -WorkingDirectory "F:\work\MAS" -NoNewWindow -PassThru -RedirectStandardOutput "results\26_dev${dev}_s${s}.log" -RedirectStandardError "results\26_dev${dev}_s${s}.err"
  }
}
Write-Output ("launched " + $jobs.Count + " device-control jobs (cpu x10 + cuda x10, wd=0.1)")
$jobs | Wait-Process
Write-Output "DEVICE_DONE"
