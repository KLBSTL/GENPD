param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$Configuration = 'Release',
    [string]$Platform = 'x64'
)

$ErrorActionPreference = 'Stop'

$executablePath = Join-Path $ProjectRoot ("x64\\{0}\\GenPD.exe" -f $Configuration)
if (-not (Test-Path -LiteralPath $executablePath)) {
    throw "GenPD executable was not found: $executablePath"
}

$dumpbin = Get-ChildItem -Path 'D:\vs2022\VC\Tools\MSVC' -Recurse -Filter dumpbin.exe -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -like '*Hostx64\x64\dumpbin.exe' } |
    Select-Object -First 1 -ExpandProperty FullName
if (-not $dumpbin) {
    throw 'dumpbin.exe was not found under D:\vs2022\VC\Tools\MSVC.'
}

$imports = & $dumpbin /IMPORTS $executablePath
if ($LASTEXITCODE -ne 0) {
    throw "dumpbin failed for $executablePath"
}

if ($imports -match '__glewPopDebugGroup') {
    throw 'GenPD.exe imports __glewPopDebugGroup, but the bundled glew32.dll does not export it.'
}

Write-Output 'Runtime dependency import check passed.'
