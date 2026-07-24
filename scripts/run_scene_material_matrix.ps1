[CmdletBinding()]
param(
    [string]$ProjectRoot = '',
    [string]$RunLabel = ('scene-material-' + (Get-Date -Format 'yyyyMMdd-HHmmss')),
    [string]$OutputDir = '',
    [ValidateSet('manifest', 'reference', 'evaluate', 'all')]
    [string]$Stage = 'manifest',
    [int]$ReferenceFrames = 120,
    [int]$ReferenceWarmup = 20,
    [int]$ReferenceIterations = 100,
    [int]$QualityFrames = 120,
    [int]$QualityWarmup = 20,
    [int]$TimingFrames = 300,
    [int]$TimingWarmup = 30,
    [int]$TimingRepetitions = 3,
    [int]$IterationsPerFrame = 16,
    [int]$ProcessTimeoutSeconds = 300,
    [int]$InterRunDelayMilliseconds = 1000,
    [string[]]$SolverVariants = @('gpu-gather-fusion-batched-ls-persistent'),
    [string[]]$SceneIds = @(),
    [string[]]$MeshIds = @(),
    [int[]]$StretchStiffnesses = @(),
    [int[]]$BendingStiffnesses = @(),
    [switch]$Force,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$invariant = [System.Globalization.CultureInfo]::InvariantCulture
$scriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $PSCommandPath }
if ($ProjectRoot -eq '') { $ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptDir '..')) }
$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
if ($OutputDir -eq '') { $OutputDir = Join-Path $ProjectRoot (Join-Path 'results' $RunLabel) }
$OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
$benchmarkScript = Join-Path $scriptDir 'run_benchmark.ps1'
$referenceScript = Join-Path $scriptDir 'run_reference.ps1'

foreach ($tool in @($benchmarkScript, $referenceScript)) { if (-not (Test-Path -LiteralPath $tool)) { throw "Missing tool: $tool" } }
if ($TimingRepetitions -lt 1 -or $ReferenceIterations -lt 1 -or $IterationsPerFrame -lt 1) { throw 'Iteration/repetition values must be positive.' }
if ($ProcessTimeoutSeconds -lt 1 -or $InterRunDelayMilliseconds -lt 0) { throw 'Timeout and inter-run delay values are invalid.' }

$scenes = @(
    [ordered]@{ id = 'hanging'; path = 'scenes\test_scene.xml' },
    [ordered]@{ id = 'moving-sphere'; path = 'scenes\moving_sphere_cloth.xml' }
)
$meshes = @(
    [ordered]@{ id = 'square-256x256'; width = 256; height = 256 },
    [ordered]@{ id = 'rect-512x128'; width = 512; height = 128 }
)
$stretchValues = @(40, 80, 160)
$bendingValues = @(10, 20, 40)
if ($SceneIds.Count -gt 0) {
    $unknown = @($SceneIds | Where-Object { $_ -notin @($scenes | ForEach-Object { $_.id }) })
    if ($unknown.Count -gt 0) { throw "Unknown scene id: $($unknown -join ', ')" }
    $scenes = @($scenes | Where-Object { $_.id -in $SceneIds })
}
if ($MeshIds.Count -gt 0) {
    $unknown = @($MeshIds | Where-Object { $_ -notin @($meshes | ForEach-Object { $_.id }) })
    if ($unknown.Count -gt 0) { throw "Unknown mesh id: $($unknown -join ', ')" }
    $meshes = @($meshes | Where-Object { $_.id -in $MeshIds })
}
if ($StretchStiffnesses.Count -gt 0) {
    $unknown = @($StretchStiffnesses | Where-Object { $_ -notin $stretchValues })
    if ($unknown.Count -gt 0) { throw "Unsupported stretch stiffness: $($unknown -join ', ')" }
    $stretchValues = $StretchStiffnesses
}
if ($BendingStiffnesses.Count -gt 0) {
    $unknown = @($BendingStiffnesses | Where-Object { $_ -notin $bendingValues })
    if ($unknown.Count -gt 0) { throw "Unsupported bending stiffness: $($unknown -join ', ')" }
    $bendingValues = $BendingStiffnesses
}

function Format-Double { param([double]$Value) return $Value.ToString('R', $invariant) }
function Read-Double {
    param($Value)
    $parsed = 0.0
    if ($null -eq $Value -or -not [double]::TryParse([string]$Value, [System.Globalization.NumberStyles]::Float, $invariant, [ref]$parsed)) { return $null }
    if ([double]::IsNaN($parsed) -or [double]::IsInfinity($parsed)) { return $null }
    return $parsed
}
function Get-Mean { param([double[]]$Values) if ($Values.Count -eq 0) { return $null }; return [double](($Values | Measure-Object -Average).Average) }
function Get-Std {
    param([double[]]$Values)
    if ($Values.Count -lt 2) { return 0.0 }
    $mean = Get-Mean $Values; $sumSquares = 0.0
    foreach ($value in $Values) { $sumSquares += ($value - $mean) * ($value - $mean) }
    return [math]::Sqrt($sumSquares / [double]($Values.Count - 1))
}
function Get-Percentile {
    param([double[]]$Values, [double]$Quantile)
    if ($Values.Count -eq 0) { return $null }
    $sorted = @($Values | Sort-Object); $position = ($sorted.Count - 1) * $Quantile; $lo = [math]::Floor($position); $hi = [math]::Ceiling($position)
    if ($lo -eq $hi) { return [double]$sorted[$lo] }
    return [double]$sorted[$lo] + ($position - $lo) * ([double]$sorted[$hi] - [double]$sorted[$lo])
}

function Get-CaseId {
    param($Scene, $Mesh, [int]$Stretch, [int]$Bending)
    return ('{0}-{1}-s{2}-b{3}' -f $Scene.id, $Mesh.id, $Stretch, $Bending)
}
function Get-ExtraArgs {
    param($Scene, $Mesh, [int]$Stretch, [int]$Bending)
    return @('--cloth-width', $Mesh.width, '--cloth-height', $Mesh.height, '--scene', $Scene.path,
        '--stretch-stiffness', (Format-Double $Stretch), '--bending-stiffness', (Format-Double $Bending))
}
function Test-RunArtifacts {
    param([string]$Directory, [int]$Frames, [int]$Warmup, [switch]$RequireQuality, [switch]$RequireReference)
    $metadataPath = Join-Path $Directory 'run_metadata.json'; $extendedPath = Join-Path $Directory 'frame_profile_extended.csv'; $presentationPath = Join-Path $Directory 'frame_presentation.csv'
    foreach ($path in @($metadataPath, $extendedPath, $presentationPath)) { if (-not (Test-Path -LiteralPath $path)) { return $false } }
    if ($RequireQuality -and -not (Test-Path -LiteralPath (Join-Path $Directory 'quality_metrics.csv'))) { return $false }
    if ($RequireReference -and -not (Test-Path -LiteralPath (Join-Path $Directory 'reference_checkpoints'))) { return $false }
    $metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
    if ([bool]$metadata.benchmark.no_render -or -not [bool]$metadata.benchmark.sync_gpu -or -not [bool]$metadata.benchmark.disable_vsync) { return $false }
    $extended = @(Import-Csv -LiteralPath $extendedPath | Where-Object { [int]$_.frame -ge $Warmup })
    $presentation = @(Import-Csv -LiteralPath $presentationPath | Where-Object { [int]$_.frame -ge $Warmup })
    if ($extended.Count -ne $Frames -or $presentation.Count -ne $Frames) { return $false }
    foreach ($row in $extended) { if ($row.frame_valid -ne '1' -or $row.termination_reason -ne 'none') { return $false } }
    foreach ($row in $presentation) { if ($row.rendered -ne '1' -or $row.gpu_sync_enabled -ne '1' -or $null -eq (Read-Double $row.frame_wall_ms)) { return $false } }
    return $true
}
function Invoke-ReferenceCase {
    param($Scene, $Mesh, [int]$Stretch, [int]$Bending)
    $caseId = Get-CaseId -Scene $Scene -Mesh $Mesh -Stretch $Stretch -Bending $Bending
    $directory = Join-Path $OutputDir (Join-Path 'references' $caseId)
    if (-not $Force -and (Test-RunArtifacts -Directory $directory -Frames $ReferenceFrames -Warmup $ReferenceWarmup -RequireQuality -RequireReference)) { return $directory }
    if ($DryRun) { Write-Host "[dry-run] reference $caseId"; return $directory }
    New-Item -ItemType Directory -Force -Path $directory | Out-Null
    & $referenceScript -ProjectRoot $ProjectRoot -RunLabel "$RunLabel-reference-$caseId" -Frames $ReferenceFrames -Warmup $ReferenceWarmup `
        -ReferenceIterations $ReferenceIterations -CheckpointStride 1 -OutputDir $directory -NoRender:$false -Uncapped:$true -SyncGpu -DisableVsync -RenderWidth 1600 -RenderHeight 900 `
        -ProcessTimeoutSeconds $ProcessTimeoutSeconds `
        -ExtraArgs (Get-ExtraArgs -Scene $Scene -Mesh $Mesh -Stretch $Stretch -Bending $Bending) | Out-Host
    if ($InterRunDelayMilliseconds -gt 0) { Start-Sleep -Milliseconds $InterRunDelayMilliseconds }
    if (-not (Test-RunArtifacts -Directory $directory -Frames $ReferenceFrames -Warmup $ReferenceWarmup -RequireQuality -RequireReference)) { throw "Incomplete reference run: $directory" }
    return $directory
}
function Invoke-QualityCase {
    param($Scene, $Mesh, [int]$Stretch, [int]$Bending, [string]$Variant, [string]$ReferenceDir)
    $caseId = Get-CaseId -Scene $Scene -Mesh $Mesh -Stretch $Stretch -Bending $Bending
    $directory = Join-Path $OutputDir (Join-Path 'quality' (Join-Path $Variant $caseId))
    if (-not $Force -and (Test-RunArtifacts -Directory $directory -Frames $QualityFrames -Warmup $QualityWarmup -RequireQuality)) { return $directory }
    if ($DryRun) { Write-Host "[dry-run] quality $Variant $caseId"; return $directory }
    New-Item -ItemType Directory -Force -Path $directory | Out-Null
    & $benchmarkScript -ProjectRoot $ProjectRoot -RunLabel "$RunLabel-quality-$Variant-$caseId" -Frames $QualityFrames -Warmup $QualityWarmup `
        -SolverVariant $Variant -IterationsPerFrame $IterationsPerFrame -QualityMetrics -QualityReferenceDir (Join-Path $ReferenceDir 'reference_checkpoints') `
        -QualityCheckpointStride 1 -OutputDir $directory -Uncapped:$true -SyncGpu -DisableVsync -RenderWidth 1600 -RenderHeight 900 `
        -ProcessTimeoutSeconds $ProcessTimeoutSeconds `
        -ExtraArgs (Get-ExtraArgs -Scene $Scene -Mesh $Mesh -Stretch $Stretch -Bending $Bending) | Out-Host
    if ($InterRunDelayMilliseconds -gt 0) { Start-Sleep -Milliseconds $InterRunDelayMilliseconds }
    if (-not (Test-RunArtifacts -Directory $directory -Frames $QualityFrames -Warmup $QualityWarmup -RequireQuality)) { throw "Incomplete quality run: $directory" }
    return $directory
}
function Invoke-TimingCase {
    param($Scene, $Mesh, [int]$Stretch, [int]$Bending, [string]$Variant, [int]$Repetition)
    $caseId = Get-CaseId -Scene $Scene -Mesh $Mesh -Stretch $Stretch -Bending $Bending
    $directory = Join-Path $OutputDir (Join-Path 'timing' (Join-Path $Variant (Join-Path $caseId ('rep{0:D2}' -f $Repetition))))
    if (-not $Force -and (Test-RunArtifacts -Directory $directory -Frames $TimingFrames -Warmup $TimingWarmup)) { return $directory }
    if ($DryRun) { Write-Host "[dry-run] timing $Variant $caseId rep$Repetition"; return $directory }
    New-Item -ItemType Directory -Force -Path $directory | Out-Null
    & $benchmarkScript -ProjectRoot $ProjectRoot -RunLabel "$RunLabel-timing-$Variant-$caseId-rep$Repetition" -Frames $TimingFrames -Warmup $TimingWarmup `
        -SolverVariant $Variant -IterationsPerFrame $IterationsPerFrame -OutputDir $directory -Uncapped:$true -SyncGpu -DisableVsync -RenderWidth 1600 -RenderHeight 900 `
        -ProcessTimeoutSeconds $ProcessTimeoutSeconds `
        -ExtraArgs (Get-ExtraArgs -Scene $Scene -Mesh $Mesh -Stretch $Stretch -Bending $Bending) | Out-Host
    if ($InterRunDelayMilliseconds -gt 0) { Start-Sleep -Milliseconds $InterRunDelayMilliseconds }
    if (-not (Test-RunArtifacts -Directory $directory -Frames $TimingFrames -Warmup $TimingWarmup)) { throw "Incomplete timing run: $directory" }
    return $directory
}
function Summarize-Case {
    param($Scene, $Mesh, [int]$Stretch, [int]$Bending, [string]$Variant, [string]$QualityDir, [string[]]$TimingDirs)
    $quality = @(Import-Csv -LiteralPath (Join-Path $QualityDir 'quality_metrics.csv') | Where-Object { [int]$_.frame -ge $QualityWarmup })
    $qualityExt = @(Import-Csv -LiteralPath (Join-Path $QualityDir 'frame_profile_extended.csv') | Where-Object { [int]$_.frame -ge $QualityWarmup })
    $timingRows = @(); $experimentRows = @(); $presentationRows = @()
    $frameMeans = @(); $totalMeans = @(); $optimizationMeans = @(); $transferMeans = @(); $dispatchMeans = @(); $readbackMeans = @(); $bufferMeans = @()
    foreach ($directory in $TimingDirs) {
        $timingRun = @(Import-Csv -LiteralPath (Join-Path $directory 'frame_profile.csv') | Where-Object { [int]$_.frame -ge $TimingWarmup })
        $experimentRun = @(Import-Csv -LiteralPath (Join-Path $directory 'frame_profile_experiment.csv') | Where-Object { [int]$_.frame -ge $TimingWarmup })
        $presentationRun = @(Import-Csv -LiteralPath (Join-Path $directory 'frame_presentation.csv') | Where-Object { [int]$_.frame -ge $TimingWarmup })
        $timingRows += $timingRun; $experimentRows += $experimentRun; $presentationRows += $presentationRun
        $frameMeans += Get-Mean @($presentationRun | ForEach-Object { Read-Double $_.frame_wall_ms } | Where-Object { $null -ne $_ })
        $totalMeans += Get-Mean @($timingRun | ForEach-Object { Read-Double $_.total_ms } | Where-Object { $null -ne $_ })
        $optimizationMeans += Get-Mean @($timingRun | ForEach-Object { Read-Double $_.optimization_ms } | Where-Object { $null -ne $_ })
        $transferMeans += Get-Mean @($timingRun | ForEach-Object { Read-Double $_.transfer_ms } | Where-Object { $null -ne $_ })
        $dispatchMeans += Get-Mean @($experimentRun | ForEach-Object { (Read-Double $_.gradient_dispatches) + (Read-Double $_.stats_dispatches) + (Read-Double $_.reduction_dispatches) + (Read-Double $_.xupdate_dispatches) + (Read-Double $_.descent_dispatches) } | Where-Object { $null -ne $_ })
        $readbackMeans += Get-Mean @($experimentRun | ForEach-Object { Read-Double $_.host_readbacks } | Where-Object { $null -ne $_ })
        $bufferMeans += Get-Mean @($experimentRun | ForEach-Object { Read-Double $_.tracked_buffer_bytes } | Where-Object { $null -ne $_ })
    }
    $position = @($quality | Where-Object { $_.has_reference -eq '1' } | ForEach-Object { Read-Double $_.position_rel_l2 } | Where-Object { $null -ne $_ })
    $velocity = @($quality | Where-Object { $_.has_reference -eq '1' } | ForEach-Object { Read-Double $_.velocity_rel_l2 } | Where-Object { $null -ne $_ })
    $energy = @($quality | Where-Object { $_.has_reference -eq '1' } | ForEach-Object { Read-Double $_.constraint_energy_rel_error } | Where-Object { $null -ne $_ })
    $meanStrain = @($quality | ForEach-Object { Read-Double $_.mean_stretch_strain } | Where-Object { $null -ne $_ })
    $maxStrain = @($quality | ForEach-Object { Read-Double $_.max_stretch_strain } | Where-Object { $null -ne $_ })
    $penetration = @($quality | ForEach-Object { Read-Double $_.max_penetration_depth } | Where-Object { $null -ne $_ })
    $timingMetadata = Get-Content -LiteralPath (Join-Path $TimingDirs[0] 'run_metadata.json') -Raw | ConvertFrom-Json
    [pscustomobject]@{
        scene_id = $Scene.id; mesh_id = $Mesh.id; cloth_width = $Mesh.width; cloth_height = $Mesh.height; vertex_count = $Mesh.width * $Mesh.height
        stretch_stiffness = $Stretch; bending_stiffness = $Bending; solver_variant = $Variant; timing_repetitions = $TimingDirs.Count
        quality_dir = $QualityDir; timing_dirs = ($TimingDirs -join ';'); timing_frames_per_repetition = $TimingFrames; timing_frame_samples = $presentationRows.Count
        git_commit = $timingMetadata.git_commit; gpu_name = $timingMetadata.gpu_name; nvidia_driver_version = $timingMetadata.nvidia_driver_version
        rendered_frame_wall_ms_mean = Get-Mean $frameMeans; rendered_frame_wall_ms_std = Get-Std $frameMeans
        rendered_frame_wall_ms_p95 = Get-Percentile -Values @($presentationRows | ForEach-Object { Read-Double $_.frame_wall_ms } | Where-Object { $null -ne $_ }) -Quantile 0.95
        total_ms_mean = Get-Mean $totalMeans; total_ms_std = Get-Std $totalMeans
        optimization_ms_mean = Get-Mean $optimizationMeans; optimization_ms_std = Get-Std $optimizationMeans
        transfer_ms_mean = Get-Mean $transferMeans; transfer_ms_std = Get-Std $transferMeans
        dispatches_mean = Get-Mean $dispatchMeans; dispatches_std = Get-Std $dispatchMeans
        host_readbacks_mean = Get-Mean $readbackMeans; host_readbacks_std = Get-Std $readbackMeans
        tracked_buffer_bytes_mean = Get-Mean $bufferMeans; tracked_buffer_bytes_std = Get-Std $bufferMeans
        p95_position_rel_l2 = Get-Percentile -Values $position -Quantile 0.95; p95_velocity_rel_l2 = Get-Percentile -Values $velocity -Quantile 0.95
        p95_energy_rel_error = Get-Percentile -Values $energy -Quantile 0.95; p95_mean_stretch_strain = Get-Percentile -Values $meanStrain -Quantile 0.95
        p95_max_stretch_strain = Get-Percentile -Values $maxStrain -Quantile 0.95; max_penetration_depth = if ($penetration.Count -gt 0) { ($penetration | Measure-Object -Maximum).Maximum } else { $null }
        invalid_quality_frames = @($qualityExt | Where-Object { $_.frame_valid -ne '1' -or $_.termination_reason -ne 'none' }).Count
        quality_failure_rate = if ($qualityExt.Count -gt 0) { [double](@($qualityExt | Where-Object { $_.frame_valid -ne '1' -or $_.termination_reason -ne 'none' }).Count) / [double]$qualityExt.Count } else { 1.0 }
    }
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$manifest = [ordered]@{
    protocol_version = 2; label = $RunLabel; measurement = 'rendered-end-to-end'; render_width = 1600; render_height = 900
    scenes = $scenes; meshes = $meshes; stretch_stiffnesses = $stretchValues; bending_stiffnesses = $bendingValues; solver_variants = $SolverVariants
    reference = [ordered]@{ frames = $ReferenceFrames; warmup = $ReferenceWarmup; iterations = $ReferenceIterations }
    quality = [ordered]@{ frames = $QualityFrames; warmup = $QualityWarmup; timing_quality_metrics = $false }
    timing = [ordered]@{ frames = $TimingFrames; warmup = $TimingWarmup; repetitions = $TimingRepetitions; iterations_per_frame = $IterationsPerFrame }
    process_timeout_seconds = $ProcessTimeoutSeconds; inter_run_delay_milliseconds = $InterRunDelayMilliseconds
}
[System.IO.File]::WriteAllText((Join-Path $OutputDir 'manifest.json'), ($manifest | ConvertTo-Json -Depth 8), [System.Text.UTF8Encoding]::new($false))
Write-Host "Scene/material manifest: $(Join-Path $OutputDir 'manifest.json')"
if ($Stage -eq 'manifest') { exit 0 }

$summary = @()
foreach ($scene in $scenes) { foreach ($mesh in $meshes) { foreach ($stretch in $stretchValues) { foreach ($bending in $bendingValues) {
    $referenceDir = Invoke-ReferenceCase -Scene $scene -Mesh $mesh -Stretch $stretch -Bending $bending
    if ($Stage -eq 'reference') { continue }
    foreach ($variant in $SolverVariants) {
        $qualityDir = Invoke-QualityCase -Scene $scene -Mesh $mesh -Stretch $stretch -Bending $bending -Variant $variant -ReferenceDir $referenceDir
        $timingDirs = @()
        for ($rep = 1; $rep -le $TimingRepetitions; ++$rep) { $timingDirs += Invoke-TimingCase -Scene $scene -Mesh $mesh -Stretch $stretch -Bending $bending -Variant $variant -Repetition $rep }
        if (-not $DryRun) { $summary += Summarize-Case -Scene $scene -Mesh $mesh -Stretch $stretch -Bending $bending -Variant $variant -QualityDir $qualityDir -TimingDirs $timingDirs }
    }
} } } }
if (-not $DryRun -and $Stage -ne 'reference') {
    $summary | Export-Csv -LiteralPath (Join-Path $OutputDir 'scene_material_summary.csv') -NoTypeInformation
    Write-Host "Scene/material summary: $(Join-Path $OutputDir 'scene_material_summary.csv')"
}
