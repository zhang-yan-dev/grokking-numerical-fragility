Set-Location F:\work\MAS
$env:PYTHONUTF8 = 1
$env:PYTHONIOENCODING = "utf-8"
& .\.venv\Scripts\python.exe -m py_compile scripts\24_wall_vs_wd.py
if (-not $?) { Write-Output "SYNTAX_FAIL"; exit 1 }
Write-Output "syntax OK"
$jobs = @()
# decisive cells: does the mul2add2 wall survive wd=0.1? (matched wd=0.01 control at 8%)
$cells = @(
  @("0.1", "800"),    # wd=0.1, coverage 8%  (frontier coverage)
  @("0.1", "2000"),   # wd=0.1, coverage 20% (where add2 grokked)
  @("0.01", "800")    # matched control vs frontier 0/10
)
foreach ($c in $cells) {
  $wd = $c[0]; $nt = $c[1]
  foreach ($s in 0..9) {
    $args = @("scripts\24_wall_vs_wd.py", "$wd", "$nt", "$s")
    $jobs += Start-Process -FilePath "F:\work\MAS\.venv\Scripts\python.exe" -ArgumentList $args -WorkingDirectory "F:\work\MAS" -NoNewWindow -PassThru -RedirectStandardOutput "results\24_wall_wd${wd}_n${nt}_s${s}.log" -RedirectStandardError "results\24_wall_wd${wd}_n${nt}_s${s}.err"
  }
}
Write-Output ("launched " + $jobs.Count + " wall-vs-wd jobs (mul2add2)")
$jobs | Wait-Process
Write-Output "WALL_DONE"
