param(
    [string]$ProjectRoot = '',
    [string]$RunLabel = 'smoke-ncu',
    [int]$Frames = 60,
    [int]$Warmup = 10,
    [ValidateSet('cpu-ncg', 'gpu-edge-scatter', 'gpu-gather-no-fusion', 'gpu-gather-fusion', 'gpu-gather-fusion-batched-ls', 'gpu-gather-fusion-batched-ls-persistent')]
    [string]$SolverVariant = 'gpu-gather-fusion-batched-ls-persistent',
    [string]$OutputDir = '',
    [string]$ExePath = '',
    [string]$NcuPath = 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.3.1\target\windows-desktop-win7-x64\ncu.exe',
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = 'Stop'
$ScriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $PSCommandPath }
if ($ProjectRoot -eq '') {
    $ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $ScriptDir '..'))
}

function Resolve-GenPDExe {
    param([string]$ProjectRoot, [string]$RequestedExePath)
    if ($RequestedExePath -ne '') {
        return [System.IO.Path]::GetFullPath($RequestedExePath)
    }
    $parent = Split-Path -Parent $ProjectRoot
    $candidates = @(
        (Join-Path $ProjectRoot 'x64\Release\GenPD.exe'),
        (Join-Path $parent 'x64\Release\GenPD.exe')
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return [System.IO.Path]::GetFullPath($candidate)
        }
    }
    throw "GenPD.exe not found. Build Release|x64 first or pass -ExePath."
}

$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
if ($OutputDir -eq '') {
    $OutputDir = Join-Path $ProjectRoot (Join-Path 'results' $RunLabel)
}
$OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
$ExePath = Resolve-GenPDExe -ProjectRoot $ProjectRoot -RequestedExePath $ExePath

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$reportDir = Join-Path $OutputDir 'nsight-compute'
New-Item -ItemType Directory -Force -Path $reportDir | Out-Null
$statusPath = Join-Path $reportDir 'ncu-status.txt'

if (!(Test-Path -LiteralPath $NcuPath)) {
    "Nsight Compute not found: $NcuPath" | Set-Content -LiteralPath $statusPath -Encoding ASCII
    Write-Warning (Get-Content -LiteralPath $statusPath -Raw)
    exit 0
}

$reportBase = Join-Path $reportDir ('GenPD-' + $RunLabel)
$appArgs = @(
    '--benchmark',
    '--project-root', $ProjectRoot,
    '--output-dir', $OutputDir,
    '--run-label', $RunLabel,
    '--solver-variant', $SolverVariant,
    '--frames', $Frames,
    '--warmup', $Warmup,
    '--no-render',
    '--uncapped',
    '--profile-gpu-queries'
)
if ($ExtraArgs) { $appArgs += $ExtraArgs }

$ncuArgs = @(
    '--target-processes', 'all',
    '--set', 'full',
    '--force-overwrite',
    '--export', $reportBase,
    $ExePath
) + $appArgs

Write-Host "Running Nsight Compute: $NcuPath $($ncuArgs -join ' ')"
Push-Location (Split-Path -Parent $ExePath)
try {
    & $NcuPath @ncuArgs
    $exitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

if ($exitCode -ne 0 -or !(Test-Path -LiteralPath ($reportBase + '.ncu-rep'))) {
    @(
        'Nsight Compute did not produce a CUDA kernel report.',
        'This is expected on many OpenGL GLSL compute-shader workloads because NCU primarily targets CUDA kernels.',
        "ncu exit code: $exitCode",
        "attempted report base: $reportBase"
    ) | Set-Content -LiteralPath $statusPath -Encoding ASCII
    Write-Warning (Get-Content -LiteralPath $statusPath -Raw)
    exit 0
}

"Nsight Compute report: $reportBase.ncu-rep" | Set-Content -LiteralPath $statusPath -Encoding ASCII
Write-Host (Get-Content -LiteralPath $statusPath -Raw)
