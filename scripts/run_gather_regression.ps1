[CmdletBinding()]
param(
    [string]$ProjectRoot = '',
    [string]$RunLabel = ('gather-regression-' + (Get-Date -Format 'yyyyMMdd-HHmmss')),
    [string]$OutputDir = '',
    [int]$ClothDimension = 386,
    [int[]]$IterationBudgets = @(1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64),
    [int]$Frames = 3,
    [int]$Warmup = 1,
    [int]$ProcessTimeoutSeconds = 180,
    [string[]]$SolverVariants = @(),
    [switch]$RequireAllValid,
    [switch]$Force,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$scriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $PSCommandPath }
if ($ProjectRoot -eq '') { $ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptDir '..')) }
$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
if ($OutputDir -eq '') { $OutputDir = Join-Path $ProjectRoot (Join-Path 'results' $RunLabel) }
$OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
if ($ClothDimension -lt 2 -or $Frames -lt 1 -or $Warmup -lt 0 -or $ProcessTimeoutSeconds -lt 1) { throw 'Invalid regression run parameters.' }
if ($IterationBudgets.Count -eq 0 -or @($IterationBudgets | Where-Object { $_ -lt 1 }).Count -gt 0) { throw 'IterationBudgets must be positive.' }

$benchmark = Join-Path $scriptDir 'run_benchmark.ps1'
if (-not (Test-Path -LiteralPath $benchmark)) { throw "Missing benchmark wrapper: $benchmark" }
$variants = @('gpu-gather-no-fusion', 'gpu-gather-fusion', 'gpu-gather-fusion-batched-ls', 'gpu-gather-fusion-batched-ls-persistent')
if ($SolverVariants.Count -gt 0) {
    $unknown = @($SolverVariants | Where-Object { $_ -notin $variants })
    if ($unknown.Count -gt 0) { throw "Unknown gather regression variant: $($unknown -join ', ')" }
    $variants = @($variants | Where-Object { $_ -in $SolverVariants })
}

function Test-CompletedRun {
    param([string]$Directory)
    $metadataPath = Join-Path $Directory 'run_metadata.json'
    $extendedPath = Join-Path $Directory 'frame_profile_extended.csv'
    $presentationPath = Join-Path $Directory 'frame_presentation.csv'
    if (-not (Test-Path -LiteralPath $metadataPath) -or -not (Test-Path -LiteralPath $extendedPath) -or -not (Test-Path -LiteralPath $presentationPath)) { return $false }
    $metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
    if ([bool]$metadata.benchmark.no_render -or -not [bool]$metadata.benchmark.sync_gpu -or -not [bool]$metadata.benchmark.disable_vsync) { return $false }
    $extended = @(Import-Csv -LiteralPath $extendedPath | Where-Object { [int]$_.frame -ge $Warmup })
    $presentation = @(Import-Csv -LiteralPath $presentationPath | Where-Object { [int]$_.frame -ge $Warmup })
    return $extended.Count -eq $Frames -and $presentation.Count -eq $Frames
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$manifest = [ordered]@{
    protocol_version = 1
    label = $RunLabel
    purpose = 'rendered gather-stability regression; not equal-quality performance evidence'
    scene = 'scenes\test_scene.xml'
    cloth_dimension = $ClothDimension
    variants = $variants
    iteration_budgets = $IterationBudgets
    measurement = [ordered]@{ rendered = $true; frames = $Frames; warmup = $Warmup; width = 1600; height = 900; gpu_sync = $true; disable_vsync = $true }
    process_timeout_seconds = $ProcessTimeoutSeconds
}
[System.IO.File]::WriteAllText((Join-Path $OutputDir 'manifest.json'), ($manifest | ConvertTo-Json -Depth 5), [System.Text.UTF8Encoding]::new($false))

$rows = @()
foreach ($variant in $variants) {
    foreach ($budget in $IterationBudgets) {
        $caseId = '{0}-d{1}-i{2:D2}' -f $variant, $ClothDimension, $budget
        $directory = Join-Path $OutputDir $caseId
        if (-not $Force -and (Test-CompletedRun $directory)) {
            Write-Host "[resume] $caseId"
        }
        elseif ($DryRun) {
            Write-Host "[dry-run] $caseId"
            continue
        }
        else {
            New-Item -ItemType Directory -Force -Path $directory | Out-Null
            $parameters = @{
                ProjectRoot = $ProjectRoot; RunLabel = "$RunLabel-$caseId"; OutputDir = $directory
                Frames = $Frames; Warmup = $Warmup; SolverVariant = $variant; IterationsPerFrame = $budget
                Uncapped = $true; SyncGpu = $true; DisableVsync = $true; RenderWidth = 1600; RenderHeight = 900
                ProcessTimeoutSeconds = $ProcessTimeoutSeconds
                ExtraArgs = @('--cloth-dimension', $ClothDimension, '--scene', 'scenes\test_scene.xml')
            }
            $runError = ''
            try {
                & $benchmark @parameters | Out-Host
            }
            catch {
                $runError = $_.Exception.Message
            }
            if (-not (Test-CompletedRun $directory)) {
                $metadataPath = Join-Path $directory 'run_metadata.json'
                $metadata = if (Test-Path -LiteralPath $metadataPath) { Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json } else { $null }
                $rows += [pscustomobject]@{
                    case_id = $caseId; solver_variant = $variant; cloth_dimension = $ClothDimension; iterations_per_frame = $budget
                    measured_frames = 0; valid = $false; invalid_frames = 0
                    termination_reasons = if ($runError -match 'timeout') { 'process_timeout' } else { 'incomplete_output' }
                    exploded_frames = 0; converged_frames = 0; simultaneous_converged_exploded = 0
                    git_commit = if ($metadata) { $metadata.git_commit } else { '' }; gpu_name = if ($metadata) { $metadata.gpu_name } else { '' }; nvidia_driver_version = if ($metadata) { $metadata.nvidia_driver_version } else { '' }
                    output_dir = $directory; run_error = $runError
                }
                continue
            }
        }
        $metadata = Get-Content -LiteralPath (Join-Path $directory 'run_metadata.json') -Raw | ConvertFrom-Json
        $extended = @(Import-Csv -LiteralPath (Join-Path $directory 'frame_profile_extended.csv') | Where-Object { [int]$_.frame -ge $Warmup })
        $profile = @(Import-Csv -LiteralPath (Join-Path $directory 'frame_profile.csv') | Where-Object { [int]$_.frame -ge $Warmup })
        $invalid = @($extended | Where-Object { $_.frame_valid -ne '1' -or $_.termination_reason -ne 'none' })
        $reasons = @($invalid | Select-Object -ExpandProperty termination_reason -Unique)
        $exploded = @($profile | Where-Object { $_.exploded -eq '1' }).Count
        $converged = @($profile | Where-Object { $_.converged -eq '1' }).Count
        $both = @($profile | Where-Object { $_.exploded -eq '1' -and $_.converged -eq '1' }).Count
        $rows += [pscustomobject]@{
            case_id = $caseId; solver_variant = $variant; cloth_dimension = $ClothDimension; iterations_per_frame = $budget
            measured_frames = $extended.Count; valid = ($invalid.Count -eq 0); invalid_frames = $invalid.Count
            termination_reasons = if ($reasons.Count -gt 0) { $reasons -join ';' } else { 'none' }
            exploded_frames = $exploded; converged_frames = $converged; simultaneous_converged_exploded = $both
            git_commit = $metadata.git_commit; gpu_name = $metadata.gpu_name; nvidia_driver_version = $metadata.nvidia_driver_version
            output_dir = $directory; run_error = ''
        }
    }
}
if (-not $DryRun) {
    $summaryPath = Join-Path $OutputDir 'gather_regression_summary.csv'
    $rows | Export-Csv -LiteralPath $summaryPath -NoTypeInformation
    Write-Host "Gather regression summary: $summaryPath"
    if ($RequireAllValid -and @($rows | Where-Object { -not $_.valid -or $_.simultaneous_converged_exploded -ne 0 }).Count -gt 0) {
        throw 'Gather regression contains an invalid case or contradictory converged/exploded state.'
    }
}
$global:LASTEXITCODE = 0
