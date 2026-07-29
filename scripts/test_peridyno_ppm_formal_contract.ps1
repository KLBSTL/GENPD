param(
    [string] $ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'
$runner = Join-Path $ProjectRoot 'scripts\run_peridyno_ppm_operating_point.ps1'
$formal = Join-Path $ProjectRoot 'scripts\run_peridyno_ppm_formal_operating_point.ps1'
$protocol = Join-Path $ProjectRoot 'docs\experiments\2026-07-27-peridyno-ppm-operating-point.md'
foreach ($path in @($runner, $formal, $protocol)) { if (-not (Test-Path -LiteralPath $path)) { throw "Missing formal PPM artifact: $path" } }

$runnerText = Get-Content -LiteralPath $runner -Raw
if ($runnerText -notmatch '\[switch\] \$SkipBuild' -or $runnerText -notmatch 'if \(-not \$SkipBuild\)') {
    throw 'The PPM runner lacks the reusable SkipBuild control required by formal repetitions.'
}

$formalText = Get-Content -LiteralPath $formal -Raw
foreach ($required in @(
    "[int] `$Warmup = 30", "[int] `$Frames = 300", "[int] `$Repetitions = 3",
    "paper-", "'frame_profile.csv'", "'run_metadata.json'", "final_state_finite",
    "frame_host_ms", "simulation_gpu_ms", "'operating_point_summary.csv'",
    "'per_repetition_summary.csv'", "nvidia-smi", "'rendered-end-to-end'", "not an equal-model or equal-quality ranking")) {
    if ($formalText -notmatch [regex]::Escape($required)) { throw "Formal PPM script lacks required contract element: $required" }
}
foreach ($required in @('formal_runner_sha256', 'base_runner_sha256', 'adapter_sha256', 'peridyno_patch_sha256')) {
    if ($formalText -notmatch [regex]::Escape($required)) { throw "Formal PPM script lacks required source hash: $required" }
}
if ($formalText -match '--headless' -or $formalText -match '--no-render') { throw 'Formal PPM runner must not use diagnostic no-render or headless timing.' }

$protocolText = Get-Content -LiteralPath $protocol -Raw
foreach ($required in @('Formal Rendered Operating-Point Protocol', '300 measured frames', 'three independent repetitions', '128 x 128', 'operating point only')) {
    if ($protocolText -notmatch [regex]::Escape($required)) { throw "PPM protocol lacks formal-reporting boundary: $required" }
}

Write-Host 'PeriDyno PPM formal operating-point contract passed.'
