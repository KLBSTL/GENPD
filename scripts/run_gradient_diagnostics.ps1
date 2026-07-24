[CmdletBinding()]
param(
    [string]$ProjectRoot = '',
    [string]$RunLabel = ('gradient-diagnostic-' + (Get-Date -Format 'yyyyMMdd-HHmmss')),
    [string]$OutputDir = '',
    [int[]]$ClothDimensions = @(128, 256, 386),
    [string[]]$SceneIds = @(),
    [int]$ProcessTimeoutSeconds = 180,
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
if ($ProcessTimeoutSeconds -lt 1) { throw 'ProcessTimeoutSeconds must be positive.' }
if ($ClothDimensions.Count -eq 0 -or @($ClothDimensions | Where-Object { $_ -lt 2 }).Count -gt 0) { throw 'ClothDimensions must be at least two.' }

$exe = Join-Path $ProjectRoot 'x64\Release\GenPD.exe'
if (-not (Test-Path -LiteralPath $exe)) { throw "Release executable is missing: $exe" }
$scenes = @(
    [ordered]@{ id = 'hanging'; path = 'scenes\test_scene.xml' },
    [ordered]@{ id = 'moving-sphere'; path = 'scenes\moving_sphere_cloth.xml' }
)
if ($SceneIds.Count -gt 0) {
    $unknown = @($SceneIds | Where-Object { $_ -notin @($scenes | ForEach-Object { $_.id }) })
    if ($unknown.Count -gt 0) { throw "Unknown scene id: $($unknown -join ', ')" }
    $scenes = @($scenes | Where-Object { $_.id -in $SceneIds })
}

function Set-RunMetadataEnvironment {
    $git = Get-Command git -ErrorAction SilentlyContinue
    if ($git) {
        $commit = & git -c "safe.directory=$ProjectRoot" -C $ProjectRoot rev-parse --short HEAD 2>$null
        if ($commit) { $env:GENPD_GIT_COMMIT = ($commit | Select-Object -First 1).Trim() }
    }
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($smi) {
        $driver = & nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>$null | Select-Object -First 1
        $gpu = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1
        if ($driver) { $env:GENPD_NVIDIA_DRIVER_VERSION = $driver.Trim() }
        if ($gpu) { $env:GENPD_GPU_NAME = $gpu.Trim() }
    }
}

function ConvertTo-ArgumentLine {
    param([string[]]$Arguments)
    return ($Arguments | ForEach-Object {
        if ($_ -match '[\s\"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
    }) -join ' '
}

function Test-CompletedCase {
    param([string]$Directory)
    $path = Join-Path $Directory 'gradient_verification.json'
    if (-not (Test-Path -LiteralPath $path)) { return $false }
    try {
        $record = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
        return [bool]$record.pass
    }
    catch { return $false }
}

Set-RunMetadataEnvironment
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$manifest = [ordered]@{
    protocol_version = 1
    label = $RunLabel
    diagnostic_only = $true
    rendering = 'disabled; not paper timing or quality evidence'
    cloth_dimensions = $ClothDimensions
    scenes = $scenes
    process_timeout_seconds = $ProcessTimeoutSeconds
}
[System.IO.File]::WriteAllText((Join-Path $OutputDir 'manifest.json'), ($manifest | ConvertTo-Json -Depth 5), [System.Text.UTF8Encoding]::new($false))

$summary = @()
foreach ($scene in $scenes) {
    foreach ($dimension in $ClothDimensions) {
        $caseId = '{0}-d{1}' -f $scene.id, $dimension
        $caseDir = Join-Path $OutputDir $caseId
        if (-not $Force -and (Test-CompletedCase $caseDir)) {
            Write-Host "[resume] $caseId"
        }
        elseif ($DryRun) {
            Write-Host "[dry-run] $caseId"
            continue
        }
        else {
            New-Item -ItemType Directory -Force -Path $caseDir | Out-Null
            $log = Join-Path $caseDir 'diagnostic_stdout.log'
            $err = Join-Path $caseDir 'diagnostic_stderr.log'
            $appArgs = @('--project-root', $ProjectRoot, '--output-dir', $caseDir, '--run-label', "$RunLabel-$caseId", '--no-render',
                '--solver-variant', 'gpu-gather-fusion', '--cloth-dimension', $dimension, '--scene', $scene.path,
                '--verify-cs-gradient', $caseDir)
            $process = Start-Process -FilePath $exe -ArgumentList (ConvertTo-ArgumentLine $appArgs) -WorkingDirectory (Split-Path -Parent $exe) `
                -RedirectStandardOutput $log -RedirectStandardError $err -PassThru -NoNewWindow
            if (-not $process.WaitForExit($ProcessTimeoutSeconds * 1000)) {
                $process.Kill(); $process.WaitForExit()
                throw "Gradient diagnostic timed out after $ProcessTimeoutSeconds seconds: $caseId"
            }
            if (-not (Test-CompletedCase $caseDir)) {
                throw "Gradient diagnostic did not pass or did not produce verification JSON: $caseId. Logs: $log, $err"
            }
        }
        $record = Get-Content -LiteralPath (Join-Path $caseDir 'gradient_verification.json') -Raw | ConvertFrom-Json
        $summary += [pscustomobject]@{
            case_id = $caseId; scene_id = $scene.id; cloth_dimension = $dimension; pass = [bool]$record.pass; csr_valid = [bool]$record.csr_valid
            edge_scatter_relative_l2 = [double]::Parse([string]$record.edge_scatter_relative_l2, $invariant)
            gather_relative_l2 = [double]::Parse([string]$record.gather_relative_l2, $invariant)
            edge_scatter_norm_relative_error = [double]::Parse([string]$record.edge_scatter_norm_relative_error, $invariant)
            gather_norm_relative_error = [double]::Parse([string]$record.gather_norm_relative_error, $invariant)
            edge_scatter_gdotd_relative_error = [double]::Parse([string]$record.edge_scatter_gdotd_relative_error, $invariant)
            gather_gdotd_relative_error = [double]::Parse([string]$record.gather_gdotd_relative_error, $invariant)
            verification_dir = $caseDir
        }
    }
}
if (-not $DryRun) {
    $summary | Export-Csv -LiteralPath (Join-Path $OutputDir 'gradient_verification_summary.csv') -NoTypeInformation
    Write-Host "Gradient diagnostic summary: $(Join-Path $OutputDir 'gradient_verification_summary.csv')"
}
$global:LASTEXITCODE = 0
