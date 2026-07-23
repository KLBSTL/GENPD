[CmdletBinding()]
param(
    [string]$ProjectRoot = '',
    [string]$HistoricalRunRoot = '',
    [string]$AuditRoot = ''
)

$ErrorActionPreference = 'Stop'
$invariant = [System.Globalization.CultureInfo]::InvariantCulture
$scriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $PSCommandPath }
if ($ProjectRoot -eq '') {
    $ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptDir '..'))
}
$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
if ($HistoricalRunRoot -eq '') {
    $HistoricalRunRoot = Join-Path $ProjectRoot 'results\paper-20260723-rendered'
}
$HistoricalRunRoot = [System.IO.Path]::GetFullPath($HistoricalRunRoot)
if ($AuditRoot -eq '') {
    $AuditRoot = Join-Path $ProjectRoot 'results\paper-20260723-rendered-audit'
}
$AuditRoot = [System.IO.Path]::GetFullPath($AuditRoot)

$calibrationPath = Join-Path $HistoricalRunRoot 'calibration.csv'
if (-not (Test-Path -LiteralPath $calibrationPath)) {
    throw "Missing historical calibration CSV: $calibrationPath"
}

function Test-FrameZeroExplosion {
    param([string]$ResultDir)
    $profilePath = Join-Path $ResultDir 'frame_profile.csv'
    if (-not (Test-Path -LiteralPath $profilePath)) { return $false }
    $first = Import-Csv -LiteralPath $profilePath | Select-Object -First 1
    return $null -ne $first -and $first.exploded -eq '1'
}

$rows = @(Import-Csv -LiteralPath $calibrationPath)
$summary = foreach ($group in ($rows | Group-Object scene_id, cloth_dimension, solver_variant)) {
    $cases = @($group.Group)
    $e0Cases = @($cases | Where-Object { Test-FrameZeroExplosion $_.result_dir })
    $qualified = @($cases | Where-Object { $_.qualified -eq '1' })
    $first = $cases | Select-Object -First 1
    # A selected qualified budget remains a qualified case. E0 candidates are
    # retained separately, so a partial scan failure cannot erase a valid
    # selection while an all-E0 case is still classified as invalid.
    $status = if ($qualified.Count -gt 0) { 'qualified' } else { 'invalid' }
    $reason = if ($qualified.Count -gt 0) { 'none' } elseif ($e0Cases.Count -gt 0) { 'E0: frame-0 explosion' } else { 'quality_gate_not_met' }
    [pscustomobject][ordered]@{
        scene_id = $first.scene_id
        cloth_dimension = [int]$first.cloth_dimension
        solver_variant = $first.solver_variant
        status = $status
        qualified = if ($status -eq 'qualified') { 1 } else { 0 }
        termination_reason = $reason
        candidates = $cases.Count
        e0_candidates = $e0Cases.Count
        qualified_candidates = $qualified.Count
        historical_result_root = $HistoricalRunRoot
    }
}

New-Item -ItemType Directory -Force -Path $AuditRoot | Out-Null
$validityPath = Join-Path $AuditRoot 'historical_validity.csv'
$summary | Sort-Object scene_id, cloth_dimension, solver_variant | Export-Csv -NoTypeInformation -Encoding utf8 -LiteralPath $validityPath

$manifest = [ordered]@{
    protocol_version = 1
    kind = 'historical-read-only-audit'
    source_run_root = $HistoricalRunRoot
    source_commit = 'c3e36ba'
    generated_utc = [DateTime]::UtcNow.ToString('o', $invariant)
    legacy_data_mutated = $false
    validity_csv = $validityPath
    status_contract = [ordered]@{
        qualified = 'legacy quality gate passed and no invalid frame'
        invalid = 'not valid for positive performance evidence'
        termination_reasons = @('none', 'E0: frame-0 explosion', 'quality_gate_not_met')
    }
}
$manifestPath = Join-Path $AuditRoot 'manifest.json'
[System.IO.File]::WriteAllText($manifestPath, ($manifest | ConvertTo-Json -Depth 6), [System.Text.UTF8Encoding]::new($false))

Write-Host "Historical validity table: $validityPath"
Write-Host "Historical audit manifest: $manifestPath"
