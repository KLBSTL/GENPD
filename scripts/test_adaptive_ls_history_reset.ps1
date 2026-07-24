param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'
$exe = Join-Path $ProjectRoot 'x64\Release\GenPD.exe'
if (-not (Test-Path -LiteralPath $exe)) {
    throw "GenPD executable was not found: $exe"
}

$outputDir = Join-Path $ProjectRoot (Join-Path 'results' ('test-adaptive-ls-history-reset-' + $PID))
& $exe '--project-root' $ProjectRoot '--no-render' `
    '--solver-variant' 'gpu-gather-fusion-adaptive-ls-persistent' `
    '--verify-adaptive-ls-history-resets' $outputDir
if ($LASTEXITCODE -ne 0) {
    throw "Adaptive history reset verification exited with $LASTEXITCODE."
}

$csvPath = Join-Path $outputDir 'adaptive_ls_history_reset_verification.csv'
if (-not (Test-Path -LiteralPath $csvPath)) {
    throw "Missing adaptive history reset verification CSV: $csvPath"
}

$rows = @(Import-Csv -LiteralPath $csvPath)
if ($rows.Count -ne 2) {
    throw "Expected reset and stiffness verification rows, got $($rows.Count)."
}
foreach ($event in @('reset', 'stiffness-change')) {
    $row = @($rows | Where-Object { $_.event -eq $event })
    if ($row.Count -ne 1) {
        throw "Missing verification row for $event."
    }
    if ($row[0].history_valid_before -ne '1' -or $row[0].history_valid_after -ne '0' `
        -or [double]$row[0].previous_accepted_step_after -ne 1.0 `
        -or [double]$row[0].current_batch_base_after -ne 1.0 `
        -or $row[0].generation_increased -ne '1' -or $row[0].passed -ne '1') {
        throw "Adaptive history reset contract failed for $event."
    }
}

Write-Host "Adaptive line-search history reset contract passed: $csvPath"
