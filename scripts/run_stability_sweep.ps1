param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$RunLabel = ('stability-' + (Get-Date -Format 'yyyyMMdd-HHmmss')),
    [ValidateSet('cpu-ncg', 'gpu-edge-scatter', 'gpu-gather-no-fusion', 'gpu-gather-fusion', 'gpu-gather-fusion-batched-ls', 'gpu-gather-fusion-batched-ls-persistent', 'gpu-xpbd-jacobi')]
    [string]$SolverVariant = 'gpu-gather-fusion-batched-ls-persistent',
    [double[]]$Timesteps = @(0.01665, 0.0333, 0.05),
    [double[]]$StretchStiffnesses = @(40, 80, 160),
    [double]$BendingStiffness = 20,
    [int]$ClothDimension = 128,
    [string]$Scene = 'scenes\moving_sphere_cloth.xml',
    [int]$Frames = 120,
    [int]$Warmup = 30,
    [int]$IterationsPerFrame = 0,
    [string]$OutputDir = '',
    [switch]$ProfileGpuQueries,
    [switch]$SyncGpu
)

$ErrorActionPreference = 'Stop'
$invariant = [System.Globalization.CultureInfo]::InvariantCulture
$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
if ($OutputDir -eq '') {
    $OutputDir = Join-Path $ProjectRoot (Join-Path 'results' $RunLabel)
}
$OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
$runRoot = Join-Path $OutputDir 'runs'
$summaryPath = Join-Path $OutputDir 'stability_sweep.csv'
$benchmarkScript = Join-Path $ProjectRoot 'scripts\run_benchmark.ps1'

if ($ClothDimension -lt 2) { throw 'ClothDimension must be at least 2.' }
if ($Frames -lt 1 -or $Warmup -lt 0) { throw 'Frames must be positive and Warmup must be nonnegative.' }
if (-not (Test-Path -LiteralPath $benchmarkScript)) { throw "Missing benchmark wrapper: $benchmarkScript" }
$scenePath = if ([System.IO.Path]::IsPathRooted($Scene)) { [System.IO.Path]::GetFullPath($Scene) } else { Join-Path $ProjectRoot $Scene }
if (-not (Test-Path -LiteralPath $scenePath)) { throw "Scene was not found: $scenePath" }

New-Item -ItemType Directory -Force -Path $runRoot | Out-Null
$rows = @()

function Format-ArgDouble {
    param([double]$Value)
    return $Value.ToString('R', $invariant)
}

function Format-LabelNumber {
    param([double]$Value)
    return (Format-ArgDouble $Value).Replace('-', 'm').Replace('.', 'p')
}

function Get-DoubleValue {
    param($Value)
    $parsed = 0.0
    if ($null -eq $Value -or -not [double]::TryParse([string]$Value, [System.Globalization.NumberStyles]::Float, $invariant, [ref]$parsed)) {
        return $null
    }
    return $parsed
}

function Get-Mean {
    param([object[]]$Values)
    if ($Values.Count -eq 0) { return 0.0 }
    return (($Values | Measure-Object -Average).Average)
}

function Get-MaxOrZero {
    param([object[]]$Values)
    if ($Values.Count -eq 0) { return 0.0 }
    return (($Values | Measure-Object -Maximum).Maximum)
}

foreach ($timestep in $Timesteps) {
    if ($timestep -le 0) { throw "All timestep values must be positive: $timestep" }
    foreach ($stretchStiffness in $StretchStiffnesses) {
        if ($stretchStiffness -le 0 -or $BendingStiffness -le 0) { throw 'All material stiffness values must be positive.' }

        $caseLabel = ('dt{0}-stretch{1}' -f (Format-LabelNumber $timestep), (Format-LabelNumber $stretchStiffness))
        $caseRunLabel = "$RunLabel-$caseLabel"
        $caseOutputDir = Join-Path $runRoot $caseLabel
        $extraArgs = @(
            '--timestep', (Format-ArgDouble $timestep),
            '--stretch-stiffness', (Format-ArgDouble $stretchStiffness),
            '--bending-stiffness', (Format-ArgDouble $BendingStiffness),
            '--cloth-dimension', $ClothDimension,
            '--scene', $Scene
        )

        Write-Host "Stability case: $caseLabel"
        & $benchmarkScript -ProjectRoot $ProjectRoot -RunLabel $caseRunLabel -SolverVariant $SolverVariant `
            -Frames $Frames -Warmup $Warmup -IterationsPerFrame $IterationsPerFrame -OutputDir $caseOutputDir `
            -QualityMetrics -ProfileGpuQueries:$ProfileGpuQueries -SyncGpu:$SyncGpu -ExtraArgs $extraArgs

        $profilePath = Join-Path $caseOutputDir 'frame_profile.csv'
        $qualityPath = Join-Path $caseOutputDir 'quality_metrics.csv'
        if (-not (Test-Path -LiteralPath $profilePath) -or -not (Test-Path -LiteralPath $qualityPath)) {
            throw "Stability case did not produce required profiles: $caseOutputDir"
        }

        $profileRows = @(Import-Csv -LiteralPath $profilePath | Where-Object { [int]$_.frame -ge $Warmup })
        $qualityRows = @(Import-Csv -LiteralPath $qualityPath | Where-Object { [int]$_.frame -ge $Warmup })
        if ($profileRows.Count -lt $Frames -or $qualityRows.Count -lt $Frames) {
            throw "Stability case has too few measured rows: $caseOutputDir"
        }

        $nonfiniteFrames = 0
        $explodedFrames = 0
        foreach ($row in $profileRows) {
            $invalid = $row.exploded -eq '1'
            foreach ($name in @('total_ms', 'gradient_norm', 'max_position')) {
                $value = Get-DoubleValue $row.$name
                if ($null -eq $value -or [double]::IsNaN($value) -or [double]::IsInfinity($value)) {
                    $invalid = $true
                }
            }
            if ($row.exploded -eq '1') { ++$explodedFrames }
            if ($invalid) { ++$nonfiniteFrames }
        }
        foreach ($row in $qualityRows) {
            if ($row.finite -ne '1') { ++$nonfiniteFrames }
        }

        $totalMs = @($profileRows | ForEach-Object { Get-DoubleValue $_.total_ms } | Where-Object { $null -ne $_ })
        $optimizationMs = @($profileRows | ForEach-Object { Get-DoubleValue $_.optimization_ms } | Where-Object { $null -ne $_ })
        $maxPositions = @($profileRows | ForEach-Object { Get-DoubleValue $_.max_position } | Where-Object { $null -ne $_ })
        $gradientNorms = @($profileRows | ForEach-Object { Get-DoubleValue $_.gradient_norm } | Where-Object { $null -ne $_ })
        $penetrations = @($qualityRows | ForEach-Object { Get-DoubleValue $_.max_penetration_depth } | Where-Object { $null -ne $_ })
        $failureRate = [double]$nonfiniteFrames / [double]($profileRows.Count + $qualityRows.Count)

        $rows += [pscustomobject]@{
            run_label = $caseRunLabel
            result_dir = $caseOutputDir
            solver_variant = $SolverVariant
            scene = $Scene
            timestep = Format-ArgDouble $timestep
            stretch_stiffness = Format-ArgDouble $stretchStiffness
            bending_stiffness = Format-ArgDouble $BendingStiffness
            cloth_dimension = $ClothDimension
            measured_frames = $Frames
            warmup_frames = $Warmup
            total_ms_mean = Get-Mean $totalMs
            optimization_ms_mean = Get-Mean $optimizationMs
            gradient_norm_final = if ($gradientNorms.Count -gt 0) { $gradientNorms[-1] } else { 0.0 }
            max_position = Get-MaxOrZero $maxPositions
            max_penetration_depth = Get-MaxOrZero $penetrations
            exploded_frames = $explodedFrames
            nonfinite_records = $nonfiniteFrames
            failure_rate = $failureRate
            stable = if ($nonfiniteFrames -eq 0 -and $explodedFrames -eq 0) { 1 } else { 0 }
        }
    }
}

$rows | Export-Csv -LiteralPath $summaryPath -NoTypeInformation
Write-Host "Stability sweep: $summaryPath"
Write-Host "Heatmap: powershell -ExecutionPolicy Bypass -File `"$ProjectRoot\scripts\plot_stability_heatmap.ps1`" -InputCsv `"$summaryPath`""
