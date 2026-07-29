param(
    [string]$ProjectRoot = '',
    [string]$RunLabel = 'smoke-nsys',
    [int]$Frames = 300,
    [int]$Warmup = 30,
    [ValidateSet('cpu-ncg', 'gpu-edge-scatter', 'gpu-gather-no-fusion', 'gpu-gather-fusion', 'gpu-gather-fusion-serial-ls-persistent', 'gpu-gather-fusion-batched-ls', 'gpu-gather-fusion-batched-ls-persistent', 'gpu-gather-fusion-adaptive-ls-persistent', 'gpu-xpbd-jacobi', 'gpu-xpbd-vertex-gather')]
    [string]$SolverVariant = 'gpu-gather-fusion-batched-ls-persistent',
    [string]$OutputDir = '',
    [string]$ExePath = '',
    [int]$RenderWidth = 1600,
    [int]$RenderHeight = 900,
    [string]$NsysPath = 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.3.2\target-windows-x64\nsys.exe',
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

function Set-RunMetadataEnv {
    param([string]$ProjectRoot)
    $commit = & git -c "safe.directory=$ProjectRoot" -C $ProjectRoot rev-parse --short HEAD 2>$null | Select-Object -First 1
    if ($commit) { $env:GENPD_GIT_COMMIT = $commit.Trim() }
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($nvidiaSmi) {
        $driver = & nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>$null | Select-Object -First 1
        $gpu = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1
        if ($driver) { $env:GENPD_NVIDIA_DRIVER_VERSION = $driver.Trim() }
        if ($gpu) { $env:GENPD_GPU_NAME = $gpu.Trim() }
    }
}

$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
if ($RenderWidth -lt 1 -or $RenderHeight -lt 1) { throw 'RenderWidth and RenderHeight must be positive.' }
if ($OutputDir -eq '') {
    $OutputDir = Join-Path $ProjectRoot (Join-Path 'results' $RunLabel)
}
$OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
$ExePath = Resolve-GenPDExe -ProjectRoot $ProjectRoot -RequestedExePath $ExePath
Set-RunMetadataEnv -ProjectRoot $ProjectRoot

if (!(Test-Path -LiteralPath $NsysPath)) {
    throw "Nsight Systems not found: $NsysPath"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$reportDir = Join-Path $OutputDir 'nsight-systems'
New-Item -ItemType Directory -Force -Path $reportDir | Out-Null
$reportBase = Join-Path $reportDir ('GenPD-' + $RunLabel)
$statusPath = Join-Path $reportDir 'nsys-status.txt'

$appArgs = @(
    '--benchmark',
    '--project-root', $ProjectRoot,
    '--output-dir', $OutputDir,
    '--run-label', $RunLabel,
    '--solver-variant', $SolverVariant,
    '--frames', $Frames,
    '--warmup', $Warmup,
    '--uncapped',
    '--sync-gpu',
    '--disable-vsync',
    '--render-resolution', $RenderWidth, $RenderHeight,
    '--profile-gpu-queries'
)
if ($ExtraArgs) { $appArgs += $ExtraArgs }

$nsysArgs = @(
    'profile',
    '--force-overwrite=true',
    '--trace=opengl,opengl-annotations',
    '--sample=none',
    '--output', $reportBase,
    $ExePath
) + $appArgs

Write-Host "Running Nsight Systems: $NsysPath $($nsysArgs -join ' ')"
# Nsight launches the legacy executable directly, so retain the same runtime
# DLL search path as run_benchmark.ps1.
Push-Location $ProjectRoot
try {
    & $NsysPath @nsysArgs
    $exitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

if ($exitCode -ne 0 -and !(Test-Path -LiteralPath ($reportBase + '.nsys-rep'))) {
    "Nsight Systems failed before producing a report. Exit code: $exitCode" | Set-Content -LiteralPath $statusPath -Encoding ASCII
    throw "Nsight Systems failed with exit code $exitCode"
}

@(
    "Nsight Systems exit code: $exitCode",
    "report: $reportBase.nsys-rep",
    "profile_csv: $(Join-Path $OutputDir 'frame_profile.csv')"
) | Set-Content -LiteralPath $statusPath -Encoding ASCII

if ($exitCode -ne 0) {
    Write-Warning "Nsight Systems returned $exitCode after producing a report. See: $statusPath"
}

Write-Host "Nsight Systems report base: $reportBase"
Write-Host "Profile CSV: $(Join-Path $OutputDir 'frame_profile.csv')"
