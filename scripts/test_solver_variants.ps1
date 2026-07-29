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

Push-Location $ProjectRoot
try {
    $helpText = & $executablePath --help 2>&1 | Out-String
    $helpExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
if ($helpExitCode -ne 0) {
    throw "GenPD --help failed with exit code $helpExitCode"
}

if ($helpText -notmatch '--solver-variant') {
    throw 'GenPD --help does not document --solver-variant.'
}

@(
    'cpu-ncg',
    'gpu-edge-scatter',
    'gpu-gather-no-fusion',
    'gpu-gather-fusion',
    'gpu-gather-fusion-serial-ls-persistent',
    'gpu-gather-fusion-batched-ls',
    'gpu-gather-fusion-batched-ls-persistent',
    'gpu-gather-fusion-adaptive-ls-persistent',
    'gpu-xpbd-jacobi',
    'gpu-xpbd-vertex-gather'
) | ForEach-Object {
    if ($helpText -notmatch [regex]::Escape($_)) {
        throw "GenPD --help does not document solver variant $_"
    }
}

@('shaders\\gradient_scatter.comp', 'shaders\\gradient_finalize.comp', 'shaders\\xpbd_constraints.comp', 'shaders\\xpbd_apply.comp', 'shaders\\xpbd_constraints_gather.comp', 'shaders\\xpbd_apply_gather.comp') | ForEach-Object {
    if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot $_))) {
        throw "Missing solver-variant shader: $_"
    }
}

Write-Output 'Solver variant command-line contract passed.'
