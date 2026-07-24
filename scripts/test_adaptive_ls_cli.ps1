param(
    [string]$ProjectRoot = ''
)

$ErrorActionPreference = 'Stop'
$ScriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $PSCommandPath }
if ($ProjectRoot -eq '') {
    $ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $ScriptDir '..'))
}
$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
$Runner = Join-Path $ScriptDir 'run_benchmark.ps1'
$ExePath = Join-Path $ProjectRoot 'x64\Release\GenPD.exe'

$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
$invalidOutput = & $ExePath '--benchmark' '--project-root' $ProjectRoot '--no-render' `
    '--solver-variant' 'gpu-gather-fusion-adaptive-ls-persistent' `
    '--adaptive-ls-history' 'invalid' '--frames' '1' '--warmup' '0' 2>&1
$invalidExitCode = $LASTEXITCODE
$ErrorActionPreference = $previousErrorActionPreference
if ($invalidExitCode -eq 0 -or ($invalidOutput -join "`n") -notmatch 'Unknown --adaptive-ls-history') {
    throw 'Invalid adaptive history mode was not rejected by GenPD.exe.'
}

foreach ($mode in @('none', 'iteration', 'frame')) {
    $label = "test-adaptive-ls-history-$mode"
    $output = Join-Path $ProjectRoot (Join-Path 'results' $label)
    & $Runner -ProjectRoot $ProjectRoot -RunLabel $label -OutputDir $output `
        -SolverVariant 'gpu-gather-fusion-adaptive-ls-persistent' `
        -AdaptiveLsHistory $mode -Frames 1 -Warmup 0 -NoRender $true | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "Adaptive history mode '$mode' did not complete."
    }
    $metadataPath = Join-Path $output 'run_metadata.json'
    if (-not (Test-Path -LiteralPath $metadataPath)) {
        throw "Missing metadata for adaptive history mode '$mode'."
    }
    $metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
    if ($metadata.solver_controls.adaptive_ls_history -ne $mode) {
        throw "Metadata history mode mismatch for '$mode'."
    }

    $traceLabel = "test-adaptive-ls-trace-$mode"
    $traceOutput = Join-Path $ProjectRoot (Join-Path 'results' $traceLabel)
    & $Runner -ProjectRoot $ProjectRoot -RunLabel $traceLabel -OutputDir $traceOutput `
        -SolverVariant 'gpu-gather-fusion-adaptive-ls-persistent' `
        -AdaptiveLsHistory $mode -Frames 2 -Warmup 0 -IterationsPerFrame 3 `
        -NoRender $true -ExtraArgs @('--cloth-dimension', '128', '--batched-ls-k', '4', '--profile-line-search-decisions') | Out-Host
    $tracePath = Join-Path $traceOutput 'line_search_trace.csv'
    $extendedPath = Join-Path $traceOutput 'frame_profile_extended.csv'
    if (-not (Test-Path -LiteralPath $tracePath) -or -not (Test-Path -LiteralPath $extendedPath)) {
        throw "Missing adaptive trace artifacts for '$mode'."
    }
    $trace = @(Import-Csv -LiteralPath $tracePath | Where-Object { $_.batch_id -eq '0' })
    if ($trace.Count -lt 2) { throw "Incomplete adaptive trace for '$mode'." }
    $extended = @(Import-Csv -LiteralPath $extendedPath)
    foreach ($field in @('adaptive_history_uses', 'adaptive_history_resets', 'adaptive_first_batch_accepts', 'adaptive_second_batch_accepts', 'adaptive_candidate_evaluations', 'adaptive_batch_count')) {
        if (-not ($extended[0].PSObject.Properties.Name -contains $field)) {
            throw "Missing extended profile field '$field'."
        }
    }
    if ($mode -eq 'none' -and @($trace | Where-Object { $_.history_valid_before -ne '0' }).Count -ne 0) {
        throw 'History mode none retained an accepted-step history.'
    }
    if ($mode -eq 'iteration') {
        $firstPerFrame = @($trace | Group-Object frame | ForEach-Object { $_.Group | Sort-Object {[int]$_.iteration} | Select-Object -First 1 })
        if (@($firstPerFrame | Where-Object { $_.history_valid_before -ne '0' }).Count -ne 0 -or @($trace | Where-Object { $_.iteration -gt 0 -and $_.history_valid_before -eq '1' }).Count -eq 0) {
            throw 'History mode iteration did not reset per frame and reuse within the frame.'
        }
    }
    if ($mode -eq 'frame' -and (@($trace | Where-Object { $_.history_valid_before -eq '1' }).Count -lt ($trace.Count - 1))) {
        throw 'History mode frame did not retain history across frames.'
    }
}

Write-Host 'Adaptive line-search CLI and metadata contract passed.'
