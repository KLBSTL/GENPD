[CmdletBinding()]
param(
    [string]$ProjectRoot = '',
    [string]$RunLabel = 'xpbd-energy-fusion-20260729-r1',
    [string]$RunRoot = '',
    [int]$ClothDimension = 256,
    [int]$IterationsPerFrame = 32,
    [int]$TimingFrames = 300,
    [int]$TimingWarmup = 30,
    [int]$TimingRepetitions = 3,
    [int]$EquivalenceFrames = 10,
    [int]$EquivalenceWarmup = 2,
    [int]$RenderWidth = 1600,
    [int]$RenderHeight = 900,
    [int]$ProcessTimeoutSeconds = 600,
    [string]$PythonExe = '',
    [switch]$DryRun,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$scriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $PSCommandPath }
if ($ProjectRoot -eq '') { $ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptDir '..')) }
$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
if ($RunRoot -eq '') { $RunRoot = Join-Path $ProjectRoot (Join-Path 'results' $RunLabel) }
$RunRoot = [System.IO.Path]::GetFullPath($RunRoot)

if ($ClothDimension -lt 2 -or $IterationsPerFrame -lt 1 -or $TimingFrames -lt 1 -or $TimingWarmup -lt 0 `
    -or $TimingRepetitions -lt 1 -or $EquivalenceFrames -lt 1 -or $EquivalenceWarmup -lt 0) {
    throw 'XPBD energy/fusion study arguments are invalid.'
}

$benchmarkScript = Join-Path $scriptDir 'run_benchmark.ps1'
$analysisScript = Join-Path $scriptDir 'analyze_xpbd_energy_fusion.py'
foreach ($path in @($benchmarkScript, $analysisScript)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing study input: $path" }
}

function Read-HardwareMetadata {
    $gpu = ''; $driver = ''
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $gpu = (& nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1).Trim()
        $driver = (& nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>$null | Select-Object -First 1).Trim()
    }
    return [ordered]@{ gpu_name = $gpu; nvidia_driver_version = $driver }
}

New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
$commit = (& git -c "safe.directory=$ProjectRoot" -C $ProjectRoot rev-parse HEAD 2>$null | Select-Object -First 1).Trim()
$manifest = [ordered]@{
    protocol_version = 'xpbd-energy-fusion-v1'
    label = $RunLabel
    git_commit = $commit
    hardware = Read-HardwareMetadata
    solver_variant = 'gpu-xpbd-jacobi'
    scene_path = 'scenes\moving_sphere_cloth.xml'
    cloth_dimension = $ClothDimension
    iterations_per_frame = $IterationsPerFrame
    energy_diagnostic = [ordered]@{
        rendered = $true; frames = 8; warmup = 2; timing_separate = $true
        definitions = @('implicit_euler_objective', 'raw_constraint_energy', 'inertia_energy', 'physical_potential', 'kinetic_energy', 'mechanical_total_energy')
        acceptance = [ordered]@{ cpu_gpu_objective_rel_error_max = 1.0e-3; directional_derivative_abs_error_max = 1.0e-6 }
    }
    equivalence = [ordered]@{ rendered = $true; frames = $EquivalenceFrames; warmup = $EquivalenceWarmup; exact_checkpoint_match = $true }
    timing = [ordered]@{ rendered = $true; frames = $TimingFrames; warmup = $TimingWarmup; repetitions = $TimingRepetitions; render_width = $RenderWidth; render_height = $RenderHeight; sync_gpu = $true; disable_vsync = $true; quality_readback_during_timing = $false }
    comparison = [ordered]@{ unchanged = @('scene', 'mesh', 'XPBD iteration budget', 'constraint shader', 'collision order', 'render settings'); changed = 'apply plus collision are one vertex dispatch instead of two.' }
}
[System.IO.File]::WriteAllText((Join-Path $RunRoot 'manifest.json'), ($manifest | ConvertTo-Json -Depth 8), [System.Text.UTF8Encoding]::new($false))

if ($DryRun) { Write-Host "XPBD energy/fusion dry run: $RunRoot"; exit 0 }

function Invoke-StudyRun {
    param(
        [string]$OutputDir,
        [int]$Frames,
        [int]$Warmup,
        [int]$Fuse,
        [int]$Repetition,
        [switch]$EnergyAudit,
        [switch]$ExportCheckpoints
    )
    if ((Test-Path -LiteralPath (Join-Path $OutputDir 'frame_profile.csv')) -and -not $Force) { return }
    $params = @{
        ProjectRoot = $ProjectRoot; RunLabel = "$RunLabel-$($OutputDir.Substring($RunRoot.Length).TrimStart('\'))"
        OutputDir = $OutputDir; SolverVariant = 'gpu-xpbd-jacobi'; IterationsPerFrame = $IterationsPerFrame
        Frames = $Frames; Warmup = $Warmup; Uncapped = $true; SyncGpu = $true; DisableVsync = $true
        RenderWidth = $RenderWidth; RenderHeight = $RenderHeight; ProcessTimeoutSeconds = $ProcessTimeoutSeconds
        ExtraArgs = @('--scene', 'scenes\moving_sphere_cloth.xml', '--cloth-dimension', $ClothDimension, '--xpbd-fuse-apply-collision', $Fuse)
    }
    if ($EnergyAudit) { $params.ExtraArgs += '--energy-audit' }
    if ($ExportCheckpoints) {
        $params.ReferenceExportDir = Join-Path $OutputDir 'reference_checkpoints'
        $params.QualityCheckpointStride = 1
    }
    & $benchmarkScript @params | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "XPBD study run failed: $OutputDir" }
}

Invoke-StudyRun -OutputDir (Join-Path $RunRoot 'energy_audit') -Frames 8 -Warmup 2 -Fuse 1 -Repetition 0 -EnergyAudit
foreach ($condition in @(@{ name = 'unfused'; fuse = 0 }, @{ name = 'fused'; fuse = 1 })) {
    Invoke-StudyRun -OutputDir (Join-Path $RunRoot (Join-Path 'equivalence' $condition.name)) -Frames $EquivalenceFrames -Warmup $EquivalenceWarmup -Fuse $condition.fuse -Repetition 0 -ExportCheckpoints
    for ($rep = 1; $rep -le $TimingRepetitions; ++$rep) {
        Invoke-StudyRun -OutputDir (Join-Path $RunRoot (Join-Path 'timing' (Join-Path $condition.name ('rep{0:D2}' -f $rep)))) -Frames $TimingFrames -Warmup $TimingWarmup -Fuse $condition.fuse -Repetition $rep
    }
}

if ($PythonExe -eq '') { $PythonExe = if (Test-Path -LiteralPath 'E:\Anaconda\envs\DL\python.exe') { 'E:\Anaconda\envs\DL\python.exe' } else { 'python' } }
& $PythonExe $analysisScript --run-root $RunRoot
if ($LASTEXITCODE -ne 0) { throw "XPBD energy/fusion analysis failed with exit code $LASTEXITCODE." }
Write-Host "XPBD energy/fusion study complete: $RunRoot"
