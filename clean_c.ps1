# Clean reclaimable caches on C:. Prints first, then deletes on confirmation.
# Pure ASCII on purpose: PowerShell 5.1 mis-decodes non-ASCII in BOM-less .ps1 files.

$ErrorActionPreference = "Continue"

$targets = @(
    @{ Path = "$env:LOCALAPPDATA\NVIDIA\DXCache";                          Note = "DirectX shader cache - rebuilds automatically" },
    @{ Path = "C:\ProgramData\NVIDIA Corporation\NVIDIA App\UpdateFramework"; Note = "driver 617.14 installer - already installed" },
    @{ Path = "$env:LOCALAPPDATA\uv\cache";                                Note = "uv cache - now redirected to D:\uvcache" },
    @{ Path = "$env:LOCALAPPDATA\Temp";                                    Note = "temp files - in-use files are skipped" }
)

function Get-DirSize($p) {
    if (-not (Test-Path -LiteralPath $p)) { return -1 }
    $s = (Get-ChildItem -LiteralPath $p -Recurse -Force -File -ErrorAction SilentlyContinue |
          Measure-Object -Property Length -Sum).Sum
    if ($null -eq $s) { return 0 }
    return $s
}

Write-Output ""
Write-Output "C: before:"
$c = Get-PSDrive C
Write-Output ("  free = {0} GB" -f [math]::Round($c.Free/1GB,1))
Write-Output ""

Write-Output "Will DELETE these directory CONTENTS (folders themselves are kept):"
Write-Output ""
$total = 0
foreach ($t in $targets) {
    $b = Get-DirSize $t.Path
    if ($b -lt 0) {
        Write-Output ("  [missing]  {0}" -f $t.Path)
    } else {
        $total += $b
        Write-Output ("  {0,8} MB  {1}" -f [math]::Round($b/1MB,0), $t.Path)
        Write-Output ("             -> {0}" -f $t.Note)
    }
}
Write-Output ""
Write-Output ("TOTAL reclaimable: {0} GB" -f [math]::Round($total/1GB,2))
Write-Output ""
Write-Output "NOT touched: C:\Windows\SoftwareDistribution\Download (needs admin)"
Write-Output "WARNING: deleting DXCache makes games recompile shaders on next launch."
Write-Output "         Expect a one-time stutter. Nothing is lost."
Write-Output ""

$ans = Read-Host "Type YES to delete, anything else to abort"
if ($ans -ne "YES") {
    Write-Output "aborted, nothing deleted."
    exit 0
}

Write-Output ""
foreach ($t in $targets) {
    if (-not (Test-Path -LiteralPath $t.Path)) { continue }
    Write-Output ("deleting contents of {0} ..." -f $t.Path)
    Get-ChildItem -LiteralPath $t.Path -Force -ErrorAction SilentlyContinue | ForEach-Object {
        try {
            Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction Stop
        } catch {
            Write-Output ("  skipped (in use): {0}" -f $_.Exception.Message.Split([char]10)[0])
        }
    }
}

Write-Output ""
$c = Get-PSDrive C
Write-Output ("C: after : free = {0} GB" -f [math]::Round($c.Free/1GB,1))
Write-Output "CLEAN_DONE"
