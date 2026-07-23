param(
    [string]$ProjectRoot = '',
    [string]$RunLabel = ('benchmark-' + (Get-Date -Format 'yyyyMMdd-HHmmss')),
    [int]$Frames = 300,
    [int]$Warmup = 30,
    [ValidateSet('cpu-ncg', 'gpu-edge-scatter', 'gpu-gather-no-fusion', 'gpu-gather-fusion', 'gpu-gather-fusion-batched-ls', 'gpu-gather-fusion-batched-ls-persistent')]
    [string]$SolverVariant = 'gpu-gather-fusion-batched-ls-persistent',
    [int]$IterationsPerFrame = 0,
    [string]$ReferenceExportDir = '',
    [string]$QualityReferenceDir = '',
    [int]$QualityCheckpointStride = 1,
    [switch]$QualityMetrics,
    [string]$OutputDir = '',
    [string]$ExePath = '',
    [switch]$ProfileGpuQueries,
    [switch]$SyncGpu,
    [int]$CaptureFrame = -1,
    [string]$CaptureOutput = '',
    [int]$CaptureWidth = 0,
    [int]$CaptureHeight = 0,
    [switch]$Headless,
    [bool]$NoRender = $true,
    [bool]$Uncapped = $true,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = 'Stop'
$ScriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $PSCommandPath }
if ($ProjectRoot -eq '') {
    $ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $ScriptDir '..'))
}

function Resolve-GenPDExe {
    param(
        [string]$ProjectRoot,
        [string]$RequestedExePath
    )

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

    $git = Get-Command git -ErrorAction SilentlyContinue
    if ($git) {
        $commit = & git -c "safe.directory=$ProjectRoot" -C $ProjectRoot rev-parse --short HEAD 2>$null
        if ($commit) {
            $env:GENPD_GIT_COMMIT = ($commit | Select-Object -First 1).Trim()
        }
    }

    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($nvidiaSmi) {
        $driver = & nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>$null | Select-Object -First 1
        if ($driver) {
            $env:GENPD_NVIDIA_DRIVER_VERSION = $driver.Trim()
        }

        $gpu = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1
        if ($gpu) {
            $env:GENPD_GPU_NAME = $gpu.Trim()
        }
    }
}

$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
if ($OutputDir -eq '') {
    $OutputDir = Join-Path $ProjectRoot (Join-Path 'results' $RunLabel)
}
$OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
$ExePath = Resolve-GenPDExe -ProjectRoot $ProjectRoot -RequestedExePath $ExePath

if ($CaptureFrame -lt -1) { throw 'CaptureFrame must be nonnegative or -1.' }
if (($CaptureWidth -lt 0) -or ($CaptureHeight -lt 0) -or (($CaptureWidth -eq 0) -ne ($CaptureHeight -eq 0))) {
    throw 'CaptureWidth and CaptureHeight must both be positive when specified.'
}
if ($CaptureFrame -ge 0 -and $CaptureOutput -eq '') {
    $CaptureOutput = Join-Path $OutputDir ('capture_frame_{0:D6}.png' -f $CaptureFrame)
}
if ($CaptureFrame -ge 0) {
    $NoRender = $false
    $Headless = $false
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
Set-RunMetadataEnv -ProjectRoot $ProjectRoot

$appArgs = @(
    '--benchmark',
    '--project-root', $ProjectRoot,
    '--output-dir', $OutputDir,
    '--run-label', $RunLabel,
    '--solver-variant', $SolverVariant,
    '--frames', $Frames,
    '--warmup', $Warmup
)

if ($NoRender) { $appArgs += '--no-render' }
if ($Headless) { $appArgs += '--headless' }
if ($Uncapped) { $appArgs += '--uncapped' }
if ($SyncGpu) { $appArgs += '--sync-gpu' }
if ($ProfileGpuQueries) { $appArgs += '--profile-gpu-queries' }
if ($IterationsPerFrame -gt 0) { $appArgs += @('--iterations-per-frame', $IterationsPerFrame) }
if ($ReferenceExportDir -ne '') { $appArgs += @('--reference-export-dir', [System.IO.Path]::GetFullPath($ReferenceExportDir)) }
if ($QualityReferenceDir -ne '') { $appArgs += @('--quality-reference-dir', [System.IO.Path]::GetFullPath($QualityReferenceDir)) }
if ($QualityMetrics) { $appArgs += '--quality-metrics' }
if ($QualityCheckpointStride -gt 0 -and ($ReferenceExportDir -ne '' -or $QualityReferenceDir -ne '')) {
    $appArgs += @('--quality-checkpoint-stride', $QualityCheckpointStride)
}
if ($CaptureFrame -ge 0) { $appArgs += @('--capture-frame', $CaptureFrame, '--capture-output', [System.IO.Path]::GetFullPath($CaptureOutput)) }
if ($CaptureWidth -gt 0) { $appArgs += @('--capture-resolution', $CaptureWidth, $CaptureHeight) }
if ($ExtraArgs) { $appArgs += $ExtraArgs }

$logPath = Join-Path $OutputDir 'benchmark_stdout.log'
Write-Host "Running: $ExePath $($appArgs -join ' ')"
Push-Location (Split-Path -Parent $ExePath)
try {
    & $ExePath @appArgs *>&1 | Tee-Object -FilePath $logPath
    $exitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

if ($exitCode -ne 0) {
    throw "Benchmark failed with exit code $exitCode. Log: $logPath"
}

Write-Host "Profile CSV: $(Join-Path $OutputDir 'frame_profile.csv')"
Write-Host "Experiment profile CSV: $(Join-Path $OutputDir 'frame_profile_experiment.csv')"
Write-Host "Run metadata: $(Join-Path $OutputDir 'run_metadata.json')"
if ($QualityMetrics -or $ReferenceExportDir -ne '' -or $QualityReferenceDir -ne '') {
    Write-Host "Quality metrics: $(Join-Path $OutputDir 'quality_metrics.csv')"
}
