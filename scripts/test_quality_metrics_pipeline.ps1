param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$Configuration = 'Release',
    [string]$Platform = 'x64'
)

$ErrorActionPreference = 'Stop'

$executablePath = Join-Path $ProjectRoot ("x64\\{0}\\GenPD.exe" -f $Configuration)
if (-not (Test-Path -LiteralPath $executablePath)) {
    throw "GenPD executable was not found: $executablePath"
}

$testRoot = Join-Path $ProjectRoot ("results\\test-quality-metrics-" + $PID)
$referenceOutputDir = Join-Path $testRoot 'reference-run'
$referenceCheckpointDir = Join-Path $referenceOutputDir 'checkpoints'
$comparisonOutputDir = Join-Path $testRoot 'comparison-run'
$configPath = Join-Path $ProjectRoot 'config\config.txt'
$configHashBefore = (Get-FileHash -LiteralPath $configPath -Algorithm SHA256).Hash

& $executablePath --benchmark --headless --uncapped --frames 2 --warmup 0 `
    --output-dir $referenceOutputDir --solver-variant cpu-ncg --iterations-per-frame 2 `
    --reference-export-dir $referenceCheckpointDir --quality-checkpoint-stride 1 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Reference checkpoint run failed with exit code $LASTEXITCODE"
}

@('reference_state_000000.bin', 'reference_state_000001.bin') | ForEach-Object {
    $checkpointPath = Join-Path $referenceCheckpointDir $_
    if (-not (Test-Path -LiteralPath $checkpointPath)) {
        throw "Missing reference checkpoint: $checkpointPath"
    }
}

& $executablePath --benchmark --headless --uncapped --frames 2 --warmup 0 `
    --output-dir $comparisonOutputDir --solver-variant gpu-gather-fusion-batched-ls-persistent `
    --quality-reference-dir $referenceCheckpointDir --quality-checkpoint-stride 1 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Quality comparison run failed with exit code $LASTEXITCODE"
}

$metricsPath = Join-Path $comparisonOutputDir 'quality_metrics.csv'
if (-not (Test-Path -LiteralPath $metricsPath)) {
    throw "Missing quality metrics: $metricsPath"
}

$rows = Import-Csv -LiteralPath $metricsPath
if ($rows.Count -ne 2) {
    throw "Expected two quality rows, got $($rows.Count)"
}

foreach ($row in $rows) {
    if ($row.has_reference -ne '1' -or $row.finite -ne '1' -or $row.exploded -ne '0') {
        throw "Unexpected quality row for frame $($row.frame): $($row | ConvertTo-Json -Compress)"
    }

    @('position_rel_l2', 'velocity_rel_l2', 'constraint_energy', 'max_penetration_depth') | ForEach-Object {
        $value = 0.0
        if (-not [double]::TryParse($row.$_, [ref]$value) -or [double]::IsNaN($value) -or [double]::IsInfinity($value)) {
            throw "Invalid $_ value in frame $($row.frame): '$($row.$_)'"
        }
    }
}

if (-not @($rows | Where-Object { [math]::Abs([double]$_.constraint_energy) -gt 1e-12 })) {
    throw 'Quality comparison reported only stale zero constraint energy.'
}

$configHashAfter = (Get-FileHash -LiteralPath $configPath -Algorithm SHA256).Hash
if ($configHashBefore -ne $configHashAfter) {
    throw 'CLI quality runs modified config/config.txt.'
}

Write-Output "Quality metrics pipeline passed: $testRoot"
