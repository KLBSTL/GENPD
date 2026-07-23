[CmdletBinding()]
param(
    [string]$ProjectRoot = '',
    [string]$RunLabel = 'paper-20260723',
    [string]$RunRoot = '',
    [ValidateSet('manifest', 'reference', 'calibrate', 'performance', 'stability', 'capture', 'all')]
    [string]$Stage = 'all',
    [switch]$DryRun,
    [switch]$Force,
    [switch]$ProfileGpuQueries
)

$ErrorActionPreference = 'Stop'
$invariant = [System.Globalization.CultureInfo]::InvariantCulture
$scriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $PSCommandPath }
if ($ProjectRoot -eq '') {
    $ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptDir '..'))
}
$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
if ($RunRoot -eq '') {
    $RunRoot = Join-Path $ProjectRoot (Join-Path 'results' $RunLabel)
}
$RunRoot = [System.IO.Path]::GetFullPath($RunRoot)

$benchmarkScript = Join-Path $scriptDir 'run_benchmark.ps1'
$referenceScript = Join-Path $scriptDir 'run_reference.ps1'
foreach ($tool in @($benchmarkScript, $referenceScript)) {
    if (-not (Test-Path -LiteralPath $tool)) { throw "Missing experiment tool: $tool" }
}

$scenes = @(
    [ordered]@{ id = 'hanging'; path = 'scenes\test_scene.xml' },
    [ordered]@{ id = 'moving-sphere'; path = 'scenes\moving_sphere_cloth.xml' }
)
$resolutions = @(128, 256, 386)
$variants = @(
    'cpu-ncg',
    'gpu-edge-scatter',
    'gpu-gather-no-fusion',
    'gpu-gather-fusion',
    'gpu-gather-fusion-batched-ls',
    'gpu-gather-fusion-batched-ls-persistent'
)
$candidateBudgets = @(1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32)
$qualityFrames = 120
$qualityWarmup = 20
$qualityCheckpointStride = 10
$qualityThreshold = 1e-3
$performanceFrames = 300
$performanceWarmup = 30
$performanceRepetitions = 3
$captureFrame = 180

function Convert-ToInvariantDouble {
    param($Value)
    $result = 0.0
    if ($null -eq $Value -or -not [double]::TryParse([string]$Value, [System.Globalization.NumberStyles]::Float, $invariant, [ref]$result)) {
        return $null
    }
    return $result
}

function Get-Percentile {
    param([double[]]$Values, [double]$Percentile)
    if ($Values.Count -eq 0) { return $null }
    $sorted = @($Values | Sort-Object)
    $position = ($sorted.Count - 1) * $Percentile
    $lower = [math]::Floor($position)
    $upper = [math]::Ceiling($position)
    if ($lower -eq $upper) { return [double]$sorted[$lower] }
    return [double]$sorted[$lower] + ($position - $lower) * ([double]$sorted[$upper] - [double]$sorted[$lower])
}

function Format-Number {
    param([double]$Value)
    return $Value.ToString('R', $invariant)
}

function Get-ShortCommit {
    $git = Get-Command git -ErrorAction SilentlyContinue
    if (-not $git) { return '' }
    $commit = & git -c "safe.directory=$ProjectRoot" -C $ProjectRoot rev-parse --short HEAD 2>$null | Select-Object -First 1
    if (-not $commit) { return '' }
    return $commit.Trim()
}

function Write-Json {
    param([string]$Path, $Value)
    $parent = Split-Path -Parent $Path
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    [System.IO.File]::WriteAllText($Path, ($Value | ConvertTo-Json -Depth 10), [System.Text.UTF8Encoding]::new($false))
}

function Write-Manifest {
    $manifestPath = Join-Path $RunRoot 'manifest.json'
    if ((Test-Path -LiteralPath $manifestPath) -and -not $Force) { return $manifestPath }
    $currentCommit = Get-ShortCommit
    $manifest = [ordered]@{
        protocol_version = 1
        label = $RunLabel
        created_utc = [DateTime]::UtcNow.ToString('o', $invariant)
        git_commit = $currentCommit
        project_root = $ProjectRoot
        resolutions = $resolutions
        scenes = $scenes
        variants = $variants
        quality_target = [ordered]@{
            reference_variant = 'cpu-ncg'
            reference_iterations_per_frame = 100
            frames = $qualityFrames
            warmup_frames = $qualityWarmup
            checkpoint_stride = $qualityCheckpointStride
            position_rel_l2_p95 = $qualityThreshold
        }
        performance = [ordered]@{
            frames = $performanceFrames
            warmup_frames = $performanceWarmup
            repetitions = $performanceRepetitions
            quality_metrics_during_timing = $false
        }
        stability = [ordered]@{
            solver_variant = 'gpu-gather-fusion-batched-ls-persistent'
            cloth_dimension = 256
            scene = 'scenes\moving_sphere_cloth.xml'
            timesteps = @((1.0 / 60.0), 0.0333, 0.05)
            stretch_stiffnesses = @(40, 80, 160)
            bending_stiffness = 20
            frames = $performanceFrames
            warmup_frames = $performanceWarmup
            repetitions = $performanceRepetitions
        }
        captures = [ordered]@{
            frame = $captureFrame
            width = 1600
            height = 900
            solver_variant = 'gpu-gather-fusion-batched-ls-persistent'
        }
    }
    Write-Json -Path $manifestPath -Value $manifest
    return $manifestPath
}

function Get-CaseExtraArgs {
    param([int]$ClothDimension, [string]$ScenePath, [double]$Timestep = 0.0, [double]$Stretch = 0.0, [double]$Bending = 0.0)
    $result = @('--cloth-dimension', $ClothDimension, '--scene', $ScenePath)
    if ($Timestep -gt 0) { $result += @('--timestep', (Format-Number $Timestep)) }
    if ($Stretch -gt 0) { $result += @('--stretch-stiffness', (Format-Number $Stretch)) }
    if ($Bending -gt 0) { $result += @('--bending-stiffness', (Format-Number $Bending)) }
    return $result
}

function Test-RunComplete {
    param([string]$OutputDir, [int]$Frames, [int]$Warmup, [switch]$RequireQuality)
    $profilePath = Join-Path $OutputDir 'frame_profile.csv'
    $metadataPath = Join-Path $OutputDir 'run_metadata.json'
    if (-not (Test-Path -LiteralPath $profilePath) -or -not (Test-Path -LiteralPath $metadataPath)) { return $false }
    $measured = @(Import-Csv -LiteralPath $profilePath | Where-Object { [int]$_.frame -ge $Warmup })
    if ($measured.Count -ne $Frames) { return $false }
    foreach ($row in $measured) {
        if ($row.exploded -eq '1') { return $false }
        foreach ($field in @('total_ms', 'optimization_ms', 'gradient_norm', 'max_position')) {
            $number = Convert-ToInvariantDouble $row.$field
            if ($null -eq $number -or [double]::IsNaN($number) -or [double]::IsInfinity($number)) { return $false }
        }
    }
    if ($RequireQuality) {
        $qualityPath = Join-Path $OutputDir 'quality_metrics.csv'
        if (-not (Test-Path -LiteralPath $qualityPath)) { return $false }
    }
    return $true
}

function Invoke-Reference {
    param([hashtable]$Scene, [int]$ClothDimension)
    $caseId = '{0}-d{1}' -f $Scene.id, $ClothDimension
    $outputDir = Join-Path $RunRoot (Join-Path 'references' $caseId)
    if (-not $Force -and (Test-RunComplete -OutputDir $outputDir -Frames $qualityFrames -Warmup $qualityWarmup -RequireQuality)) { return $outputDir }
    if ($DryRun) { Write-Host "[dry-run] reference $caseId"; return $outputDir }
    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
    & $referenceScript -ProjectRoot $ProjectRoot -RunLabel ("$RunLabel-reference-$caseId") `
        -Frames $qualityFrames -Warmup $qualityWarmup -ReferenceIterations 100 -CheckpointStride $qualityCheckpointStride `
        -OutputDir $outputDir -NoRender:$true -Uncapped:$true `
        -ExtraArgs (Get-CaseExtraArgs -ClothDimension $ClothDimension -ScenePath $Scene.path)
    if (-not (Test-RunComplete -OutputDir $outputDir -Frames $qualityFrames -Warmup $qualityWarmup -RequireQuality)) {
        throw "Reference run is incomplete or invalid: $outputDir"
    }
    return $outputDir
}

function Get-QualitySummary {
    param([string]$OutputDir)
    $qualityPath = Join-Path $OutputDir 'quality_metrics.csv'
    $profilePath = Join-Path $OutputDir 'frame_profile.csv'
    if (-not (Test-Path -LiteralPath $qualityPath) -or -not (Test-Path -LiteralPath $profilePath)) {
        throw "Missing quality/profile output: $OutputDir"
    }
    $qualityRows = @(Import-Csv -LiteralPath $qualityPath | Where-Object { [int]$_.frame -ge $qualityWarmup })
    $profileRows = @(Import-Csv -LiteralPath $profilePath | Where-Object { [int]$_.frame -ge $qualityWarmup })
    $invalid = 0
    $referenceRows = @()
    foreach ($row in $qualityRows) {
        if ($row.finite -ne '1' -or $row.exploded -eq '1') { ++$invalid; continue }
        if ($row.has_reference -eq '1') { $referenceRows += $row }
    }
    foreach ($row in $profileRows) {
        if ($row.exploded -eq '1') { ++$invalid }
    }
    $position = @($referenceRows | ForEach-Object { Convert-ToInvariantDouble $_.position_rel_l2 } | Where-Object { $null -ne $_ })
    $velocity = @($referenceRows | ForEach-Object { Convert-ToInvariantDouble $_.velocity_rel_l2 } | Where-Object { $null -ne $_ })
    $energy = @($referenceRows | ForEach-Object { Convert-ToInvariantDouble $_.constraint_energy_rel_error } | Where-Object { $null -ne $_ })
    $penetration = @($qualityRows | ForEach-Object { Convert-ToInvariantDouble $_.max_penetration_depth } | Where-Object { $null -ne $_ })
    return [pscustomobject]@{
        reference_rows = $referenceRows.Count
        invalid_records = $invalid
        p95_position_rel_l2 = Get-Percentile -Values $position -Percentile 0.95
        p95_velocity_rel_l2 = Get-Percentile -Values $velocity -Percentile 0.95
        p95_energy_rel_error = Get-Percentile -Values $energy -Percentile 0.95
        max_penetration_depth = if ($penetration.Count -gt 0) { ($penetration | Measure-Object -Maximum).Maximum } else { $null }
        failure_rate = if (($qualityRows.Count + $profileRows.Count) -gt 0) { [double]$invalid / [double]($qualityRows.Count + $profileRows.Count) } else { 1.0 }
    }
}

function Invoke-Calibration {
    $rows = @()
    foreach ($scene in $scenes) {
        foreach ($dimension in $resolutions) {
            $referenceDir = Invoke-Reference -Scene $scene -ClothDimension $dimension
            $checkpointDir = Join-Path $referenceDir 'reference_checkpoints'
            if (-not $DryRun -and -not (Test-Path -LiteralPath $checkpointDir)) { throw "Missing reference checkpoints: $checkpointDir" }
            foreach ($variant in $variants) {
                $selected = $null
                foreach ($budget in $candidateBudgets) {
                    $caseId = '{0}-d{1}-{2}-i{3:D2}' -f $scene.id, $dimension, $variant, $budget
                    $outputDir = Join-Path $RunRoot (Join-Path 'calibration' $caseId)
                    if ($DryRun) {
                        Write-Host "[dry-run] calibration $caseId"
                        continue
                    }
                    if ($Force -or -not (Test-RunComplete -OutputDir $outputDir -Frames $qualityFrames -Warmup $qualityWarmup -RequireQuality)) {
                        New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
                        & $benchmarkScript -ProjectRoot $ProjectRoot -RunLabel ("$RunLabel-calibration-$caseId") `
                            -Frames $qualityFrames -Warmup $qualityWarmup -SolverVariant $variant -IterationsPerFrame $budget `
                            -QualityReferenceDir $checkpointDir -QualityCheckpointStride $qualityCheckpointStride `
                            -OutputDir $outputDir -NoRender:$true -Uncapped:$true `
                            -ExtraArgs (Get-CaseExtraArgs -ClothDimension $dimension -ScenePath $scene.path)
                    }
                    $metrics = Get-QualitySummary -OutputDir $outputDir
                    $qualified = $metrics.reference_rows -gt 0 -and $metrics.invalid_records -eq 0 -and `
                        $null -ne $metrics.p95_position_rel_l2 -and $metrics.p95_position_rel_l2 -le $qualityThreshold
                    $row = [pscustomobject]@{
                        scene_id = $scene.id
                        scene_path = $scene.path
                        cloth_dimension = $dimension
                        solver_variant = $variant
                        iterations_per_frame = $budget
                        qualified = [int]$qualified
                        reference_rows = $metrics.reference_rows
                        p95_position_rel_l2 = $metrics.p95_position_rel_l2
                        p95_velocity_rel_l2 = $metrics.p95_velocity_rel_l2
                        p95_energy_rel_error = $metrics.p95_energy_rel_error
                        max_penetration_depth = $metrics.max_penetration_depth
                        failure_rate = $metrics.failure_rate
                        result_dir = $outputDir
                        reference_dir = $referenceDir
                    }
                    $rows += $row
                    if ($qualified -and $null -eq $selected) { $selected = $row }
                }
            }
        }
    }
    if ($DryRun) { return }
    $calibrationPath = Join-Path $RunRoot 'calibration.csv'
    $rows | Export-Csv -LiteralPath $calibrationPath -NoTypeInformation
    $selectedRows = @($rows | Group-Object scene_id, cloth_dimension, solver_variant | ForEach-Object {
        $_.Group | Where-Object { $_.qualified -eq 1 } | Sort-Object { [int]$_.iterations_per_frame } | Select-Object -First 1
    } | Where-Object { $null -ne $_ })
    $selectedPath = Join-Path $RunRoot 'selected_budgets.csv'
    $selectedRows | Export-Csv -LiteralPath $selectedPath -NoTypeInformation
    if ($selectedRows.Count -eq 0) { throw 'No calibration case met the equal-quality threshold.' }
    Write-Host "Calibration candidates: $calibrationPath"
    Write-Host "Selected equal-quality budgets: $selectedPath"
}

function Invoke-Performance {
    $selectedPath = Join-Path $RunRoot 'selected_budgets.csv'
    if (-not (Test-Path -LiteralPath $selectedPath)) { throw "Missing calibration selection: $selectedPath. Run -Stage calibrate first." }
    $selected = @(Import-Csv -LiteralPath $selectedPath)
    foreach ($row in $selected) {
        for ($rep = 1; $rep -le $performanceRepetitions; ++$rep) {
            $caseId = '{0}-d{1}-{2}-rep{3:D2}' -f $row.scene_id, $row.cloth_dimension, $row.solver_variant, $rep
            $outputDir = Join-Path $RunRoot (Join-Path 'performance' $caseId)
            if ($DryRun) { Write-Host "[dry-run] performance $caseId"; continue }
            if (-not $Force -and (Test-RunComplete -OutputDir $outputDir -Frames $performanceFrames -Warmup $performanceWarmup)) { continue }
            New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
            & $benchmarkScript -ProjectRoot $ProjectRoot -RunLabel ("$RunLabel-performance-$caseId") `
                -Frames $performanceFrames -Warmup $performanceWarmup -SolverVariant $row.solver_variant `
                -IterationsPerFrame ([int]$row.iterations_per_frame) -OutputDir $outputDir `
                -NoRender:$true -Uncapped:$true -ProfileGpuQueries:$ProfileGpuQueries `
                -ExtraArgs (Get-CaseExtraArgs -ClothDimension ([int]$row.cloth_dimension) -ScenePath $row.scene_path)
            if (-not (Test-RunComplete -OutputDir $outputDir -Frames $performanceFrames -Warmup $performanceWarmup)) {
                throw "Performance run is incomplete or invalid: $outputDir"
            }
        }
    }
}

function Invoke-Stability {
    $rows = @()
    foreach ($dt in @((1.0 / 60.0), 0.0333, 0.05)) {
        foreach ($stretch in @(40.0, 80.0, 160.0)) {
            for ($rep = 1; $rep -le $performanceRepetitions; ++$rep) {
                $dtLabel = (Format-Number $dt).Replace('.', 'p')
                $caseId = 'dt{0}-stretch{1}-rep{2:D2}' -f $dtLabel, $stretch, $rep
                $outputDir = Join-Path $RunRoot (Join-Path 'stability' $caseId)
                if ($DryRun) { Write-Host "[dry-run] stability $caseId"; continue }
                if ($Force -or -not (Test-RunComplete -OutputDir $outputDir -Frames $performanceFrames -Warmup $performanceWarmup -RequireQuality)) {
                    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
                    & $benchmarkScript -ProjectRoot $ProjectRoot -RunLabel ("$RunLabel-stability-$caseId") `
                        -Frames $performanceFrames -Warmup $performanceWarmup -SolverVariant 'gpu-gather-fusion-batched-ls-persistent' `
                        -OutputDir $outputDir -NoRender:$true -Uncapped:$true -QualityMetrics `
                        -ExtraArgs (Get-CaseExtraArgs -ClothDimension 256 -ScenePath 'scenes\moving_sphere_cloth.xml' -Timestep $dt -Stretch $stretch -Bending 20.0)
                }
                $profileRows = @(Import-Csv -LiteralPath (Join-Path $outputDir 'frame_profile.csv') | Where-Object { [int]$_.frame -ge $performanceWarmup })
                $qualityRows = @(Import-Csv -LiteralPath (Join-Path $outputDir 'quality_metrics.csv') | Where-Object { [int]$_.frame -ge $performanceWarmup })
                $invalid = @($profileRows | Where-Object { $_.exploded -eq '1' }).Count + @($qualityRows | Where-Object { $_.finite -ne '1' -or $_.exploded -eq '1' }).Count
                $rows += [pscustomobject]@{
                    timestep = Format-Number $dt
                    stretch_stiffness = Format-Number $stretch
                    bending_stiffness = 20
                    repetition = $rep
                    stable = [int]($invalid -eq 0)
                    invalid_records = $invalid
                    result_dir = $outputDir
                }
            }
        }
    }
    if ($DryRun) { return }
    $replicatePath = Join-Path $RunRoot 'stability_replicates.csv'
    $rows | Export-Csv -LiteralPath $replicatePath -NoTypeInformation
    $summary = @($rows | Group-Object timestep, stretch_stiffness | ForEach-Object {
        $group = $_.Group
        [pscustomobject]@{
            timestep = $group[0].timestep
            stretch_stiffness = $group[0].stretch_stiffness
            repetitions = $group.Count
            stable_repetitions = @($group | Where-Object { $_.stable -eq 1 }).Count
            stable = [int](@($group | Where-Object { $_.stable -eq 1 }).Count -eq $performanceRepetitions)
        }
    })
    $summary | Export-Csv -LiteralPath (Join-Path $RunRoot 'stability_summary.csv') -NoTypeInformation
}

function Invoke-Captures {
    $selectedPath = Join-Path $RunRoot 'selected_budgets.csv'
    $selected = if (Test-Path -LiteralPath $selectedPath) { @(Import-Csv -LiteralPath $selectedPath) } else { @() }
    foreach ($scene in $scenes) {
        foreach ($dimension in $resolutions) {
            $matching = @($selected | Where-Object { $_.scene_id -eq $scene.id -and [int]$_.cloth_dimension -eq $dimension -and $_.solver_variant -eq 'gpu-gather-fusion-batched-ls-persistent' })
            $iterations = if ($matching.Count -eq 1) { [int]$matching[0].iterations_per_frame } else { 32 }
            $caseId = '{0}-d{1}' -f $scene.id, $dimension
            $outputDir = Join-Path $RunRoot (Join-Path 'captures' $caseId)
            $capturePath = Join-Path $outputDir 'frame-180.png'
            if (-not $Force -and (Test-Path -LiteralPath $capturePath)) { continue }
            if ($DryRun) { Write-Host "[dry-run] capture $caseId"; continue }
            New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
            & $benchmarkScript -ProjectRoot $ProjectRoot -RunLabel ("$RunLabel-capture-$caseId") `
                -Frames ($captureFrame + 1) -Warmup 0 -SolverVariant 'gpu-gather-fusion-batched-ls-persistent' `
                -IterationsPerFrame $iterations -OutputDir $outputDir -NoRender:$false -Uncapped:$true `
                -CaptureFrame $captureFrame -CaptureOutput $capturePath -CaptureWidth 1600 -CaptureHeight 900 `
                -ExtraArgs (Get-CaseExtraArgs -ClothDimension $dimension -ScenePath $scene.path)
            if (-not (Test-Path -LiteralPath $capturePath)) { throw "Screenshot was not produced: $capturePath" }
        }
    }
}

New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
$manifestPath = Write-Manifest
Write-Host "Paper experiment manifest: $manifestPath"
if ($Stage -eq 'manifest') { exit 0 }

if ($Stage -eq 'reference' -or $Stage -eq 'all') {
    foreach ($scene in $scenes) { foreach ($dimension in $resolutions) { [void](Invoke-Reference -Scene $scene -ClothDimension $dimension) } }
}
if ($Stage -eq 'calibrate' -or $Stage -eq 'all') { Invoke-Calibration }
if ($Stage -eq 'performance' -or $Stage -eq 'all') { Invoke-Performance }
if ($Stage -eq 'stability' -or $Stage -eq 'all') { Invoke-Stability }
if ($Stage -eq 'capture' -or $Stage -eq 'all') { Invoke-Captures }
