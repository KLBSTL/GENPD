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

$help = (& $executablePath --help 2>&1) -join "`n"
@('--capture-frame N', '--capture-output PATH', '--capture-resolution W H') | ForEach-Object {
    if ($help -notmatch [regex]::Escape($_)) {
        throw "Missing capture option in --help: $_"
    }
}

$outputDir = Join-Path $ProjectRoot ("results\\test-capture-" + $PID)
$capturePath = Join-Path $outputDir 'capture.png'
& $executablePath --benchmark --uncapped --frames 1 --warmup 0 --project-root $ProjectRoot `
    --output-dir $outputDir --run-label 'test-capture' --cloth-dimension 16 `
    --capture-frame 0 --capture-output $capturePath --capture-resolution 320 240 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $capturePath)) {
    throw 'Capture run did not produce the requested screenshot.'
}

$bytes = [IO.File]::ReadAllBytes($capturePath)
if ($bytes.Length -le 1024 -or $bytes[0..7] -join ',' -ne '137,80,78,71,13,10,26,10') {
    throw 'Capture output is not a nonempty PNG file.'
}

Write-Output "Capture CLI contract passed: $capturePath"
