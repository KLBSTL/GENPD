param(
    [string]$ProjectRoot = '',
    [string]$RunLabel = ('benchmark-' + (Get-Date -Format 'yyyyMMdd-HHmmss')),
    [int]$Frames = 300,
    [int]$Warmup = 30,
    [ValidateSet('cpu-ncg', 'gpu-edge-scatter', 'gpu-gather-no-fusion', 'gpu-gather-fusion', 'gpu-gather-fusion-batched-ls', 'gpu-gather-fusion-batched-ls-persistent', 'gpu-gather-fusion-adaptive-ls-persistent', 'gpu-xpbd-jacobi')]
    [string]$SolverVariant = 'gpu-gather-fusion-batched-ls-persistent',
    [int]$IterationsPerFrame = 0,
    [string]$ReferenceExportDir = '',
    [string]$QualityReferenceDir = '',
    [int]$QualityCheckpointStride = 1,
    [switch]$QualityMetrics,
    [string]$OutputDir = '',
    [string]$ExePath = '',
    [switch]$ProfileGpuQueries,
    [ValidateSet('none', 'iteration', 'frame')]
    [string]$AdaptiveLsHistory = 'frame',
    [switch]$ForceCpuStateRoundtrip,
    [switch]$SyncGpu,
    [switch]$DisableVsync,
    [int]$CaptureFrame = -1,
    [string]$CaptureOutput = '',
    [int]$CaptureWidth = 0,
    [int]$CaptureHeight = 0,
    [int]$RenderWidth = 0,
    [int]$RenderHeight = 0,
    [int]$ProcessTimeoutSeconds = 0,
    [switch]$Headless,
    [bool]$NoRender = $false,
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
if (($RenderWidth -lt 0) -or ($RenderHeight -lt 0) -or (($RenderWidth -eq 0) -ne ($RenderHeight -eq 0))) {
    throw 'RenderWidth and RenderHeight must both be positive when specified.'
}
if ($ProcessTimeoutSeconds -lt 0) { throw 'ProcessTimeoutSeconds must be nonnegative.' }
if ($CaptureFrame -ge 0 -and $CaptureOutput -eq '') {
    $CaptureOutput = Join-Path $OutputDir ('capture_frame_{0:D6}.png' -f $CaptureFrame)
}
if ($CaptureFrame -ge 0) {
    $NoRender = $false
    $Headless = $false
}
if ($NoRender -and $RunLabel -match '^paper-') {
    throw 'Paper-labelled runs must use rendered measurements; --no-render is reserved for diagnostics and short regressions.'
}
if ($ForceCpuStateRoundtrip -and $SolverVariant -notin @('gpu-gather-fusion-batched-ls-persistent', 'gpu-gather-fusion-adaptive-ls-persistent')) {
    throw 'ForceCpuStateRoundtrip is defined only for fixed or adaptive persistent gather-fusion NCG.'
}
if ($ForceCpuStateRoundtrip -and $RunLabel -match '^paper-') {
    throw 'ForceCpuStateRoundtrip is a diagnostic counterfactual and cannot use a paper-labelled run label.'
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
Set-RunMetadataEnv -ProjectRoot $ProjectRoot

$appArgs = @(
    '--benchmark',
    '--project-root', $ProjectRoot,
    '--output-dir', $OutputDir,
    '--run-label', $RunLabel,
    '--solver-variant', $SolverVariant,
    '--adaptive-ls-history', $AdaptiveLsHistory,
    '--frames', $Frames,
    '--warmup', $Warmup
)

if ($NoRender) { $appArgs += '--no-render' }
if ($Headless) { $appArgs += '--headless' }
if ($Uncapped) { $appArgs += '--uncapped' }
if ($SyncGpu) { $appArgs += '--sync-gpu' }
if ($DisableVsync) { $appArgs += '--disable-vsync' }
if ($ProfileGpuQueries) { $appArgs += '--profile-gpu-queries' }
if ($ForceCpuStateRoundtrip) { $appArgs += '--force-cpu-state-roundtrip' }
if ($IterationsPerFrame -gt 0) { $appArgs += @('--iterations-per-frame', $IterationsPerFrame) }
if ($ReferenceExportDir -ne '') { $appArgs += @('--reference-export-dir', [System.IO.Path]::GetFullPath($ReferenceExportDir)) }
if ($QualityReferenceDir -ne '') { $appArgs += @('--quality-reference-dir', [System.IO.Path]::GetFullPath($QualityReferenceDir)) }
if ($QualityMetrics) { $appArgs += '--quality-metrics' }
if ($QualityCheckpointStride -gt 0 -and ($ReferenceExportDir -ne '' -or $QualityReferenceDir -ne '')) {
    $appArgs += @('--quality-checkpoint-stride', $QualityCheckpointStride)
}
if ($CaptureFrame -ge 0) { $appArgs += @('--capture-frame', $CaptureFrame, '--capture-output', [System.IO.Path]::GetFullPath($CaptureOutput)) }
if ($CaptureWidth -gt 0) { $appArgs += @('--capture-resolution', $CaptureWidth, $CaptureHeight) }
if ($RenderWidth -gt 0) { $appArgs += @('--render-resolution', $RenderWidth, $RenderHeight) }
if ($ExtraArgs) { $appArgs += $ExtraArgs }

$logPath = Join-Path $OutputDir 'benchmark_stdout.log'
Write-Host "Running: $ExePath $($appArgs -join ' ')"
# Legacy OpenGL runtime DLLs are part of the project root, while the Release
# executable lives below x64\Release. Keep the process working directory at
# the root so interactive, benchmark, and profiler launches resolve alike.
Push-Location $ProjectRoot
try {
    if ($ProcessTimeoutSeconds -eq 0) {
        & $ExePath @appArgs *>&1 | Tee-Object -FilePath $logPath
        $exitCode = $LASTEXITCODE
    }
    else {
        $stderrPath = Join-Path $OutputDir 'benchmark_stderr.log'
        $argumentLine = ($appArgs | ForEach-Object {
            if ($_ -match '[\s\"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
        }) -join ' '
        $process = Start-Process -FilePath $ExePath -ArgumentList $argumentLine -WorkingDirectory $ProjectRoot `
            -RedirectStandardOutput $logPath -RedirectStandardError $stderrPath -PassThru -NoNewWindow
        if (-not $process.WaitForExit($ProcessTimeoutSeconds * 1000)) {
            $process.Kill()
            $process.WaitForExit()
            throw "Benchmark exceeded the $ProcessTimeoutSeconds second timeout. Logs: $logPath, $stderrPath"
        }
        $process.WaitForExit()
        $process.Refresh()
        $exitCode = $process.ExitCode
        # Preserve complete child logs on disk, but keep a long paper matrix
        # readable in the terminal.
        Get-Content -LiteralPath $logPath -Tail 80 -ErrorAction SilentlyContinue | Write-Host
        Get-Content -LiteralPath $stderrPath -Tail 80 -ErrorAction SilentlyContinue | Write-Host
        if ($null -eq $exitCode) {
            $requiredArtifactNames = @('frame_profile.csv', 'frame_profile_experiment.csv', 'run_metadata.json')
            if (-not $NoRender -and -not $Headless) {
                $requiredArtifactNames += 'frame_presentation.csv'
            }
            $requiredArtifacts = $requiredArtifactNames | ForEach-Object { Join-Path $OutputDir $_ }
            $missingArtifacts = @($requiredArtifacts | Where-Object { -not (Test-Path -LiteralPath $_) })
            if ($missingArtifacts.Count -gt 0) {
                throw "Benchmark process ended without an observable exit code and did not produce: $($missingArtifacts -join ', ')"
            }
            $exitCode = 0
        }
    }
}
finally {
    Pop-Location
}

if ($exitCode -ne 0) {
    throw "Benchmark failed with exit code $exitCode. Log: $logPath"
}

Write-Host "Profile CSV: $(Join-Path $OutputDir 'frame_profile.csv')"
Write-Host "Experiment profile CSV: $(Join-Path $OutputDir 'frame_profile_experiment.csv')"
if (-not $NoRender -and -not $Headless) {
    Write-Host "Presentation profile CSV: $(Join-Path $OutputDir 'frame_presentation.csv')"
}
Write-Host "Run metadata: $(Join-Path $OutputDir 'run_metadata.json')"
if ($QualityMetrics -or $ReferenceExportDir -ne '' -or $QualityReferenceDir -ne '') {
    Write-Host "Quality metrics: $(Join-Path $OutputDir 'quality_metrics.csv')"
}
$global:LASTEXITCODE = 0
