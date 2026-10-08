Set-Location F:\work\MAS
$env:PYTHONUTF8 = 1
$env:PYTHONIOENCODING = "utf-8"
& .\.venv\Scripts\python.exe -m py_compile scripts\25_decomp_vs_wd.py
if (-not $?) { Write-Output "SYNTAX_FAIL"; exit 1 }
Write-Output "syntax OK"
$jobs = @()
foreach ($wd in "0.1", "0.01") {
  foreach ($s in 0..9) {
    $args = @("scripts\25_decomp_vs_wd.py", "$wd", "$s")
    $jobs += Start-Process -FilePath "F:\work\MAS\.venv\Scripts\python.exe" -ArgumentList $args -WorkingDirectory "F:\work\MAS" -NoNewWindow -PassThru -RedirectStandardOutput "results\25_decomp_wd${wd}_s${s}.log" -RedirectStandardError "results\25_decomp_wd${wd}_s${s}.err"
  }
}
Write-Output ("launched " + $jobs.Count + " decomp-vs-wd jobs (pipeline+monolith per job)")
$jobs | Wait-Process
Write-Output "DECOMP_DONE"
