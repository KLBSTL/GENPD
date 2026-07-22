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

$helpText = & $executablePath --help 2>&1 | Out-String
if ($LASTEXITCODE -ne 0) {
    throw "GenPD --help failed with exit code $LASTEXITCODE"
}

if ($helpText -notmatch '--solver-variant') {
    throw 'GenPD --help does not document --solver-variant.'
}

@(
    'cpu-ncg',
    'gpu-edge-scatter',
    'gpu-gather-no-fusion',
    'gpu-gather-fusion',
    'gpu-gather-fusion-batched-ls',
    'gpu-gather-fusion-batched-ls-persistent'
) | ForEach-Object {
    if ($helpText -notmatch [regex]::Escape($_)) {
        throw "GenPD --help does not document solver variant $_"
    }
}

@('shaders\\gradient_scatter.comp', 'shaders\\gradient_finalize.comp') | ForEach-Object {
    if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot $_))) {
        throw "Missing solver-variant shader: $_"
    }
}

Write-Output 'Solver variant command-line contract passed.'
