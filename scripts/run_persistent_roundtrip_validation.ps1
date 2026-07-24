param(
    [string]$ProjectRoot = '',
    [string]$RunLabel = ('state-roundtrip-' + (Get-Date -Format 'yyyyMMdd-HHmmss')),
    [string]$Scene = 'scenes\moving_sphere_cloth.xml',
    [int]$ClothDimension = 386,
    [int]$IterationsPerFrame = 1,
    [int]$Frames = 300,
    [int]$Warmup = 30,
    [int]$Repetitions = 3,
    [int]$RenderWidth = 1600,
    [int]$RenderHeight = 900,
    [int]$ProcessTimeoutSeconds = 600,
    [string]$PythonExe = ''
)

$ErrorActionPreference = 'Stop'
$ScriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $PSCommandPath }
if ($ProjectRoot -eq '') {
    $ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $ScriptDir '..'))
}
$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
if ($RunLabel -match '^paper-') {
    throw 'This is a diagnostic counterfactual. Use a non-paper run label.'
}
if ($ClothDimension -lt 2 -or $IterationsPerFrame -lt 1 -or $Frames -lt 1 -or $Warmup -lt 0 -or $Repetitions -lt 2) {
    throw 'ClothDimension >= 2, IterationsPerFrame >= 1, Frames >= 1, Warmup >= 0, and Repetitions >= 2 are required.'
}

$runRoot = Join-Path $ProjectRoot (Join-Path 'results' $RunLabel)
New-Item -ItemType Directory -Force -Path $runRoot | Out-Null
$manifest = [ordered]@{
    protocol = 'persistent-state-roundtrip-v1'
    purpose = 'Counterfactual rendered measurement of per-frame GPU-to-CPU state synchronization and next-frame state reupload.'
    scene = $Scene
    cloth_dimension = $ClothDimension
    solver_variant = 'gpu-gather-fusion-batched-ls-persistent'
    iterations_per_frame = $IterationsPerFrame
    frames = $Frames
    warmup_frames = $Warmup
    repetitions = $Repetitions
    render_width = $RenderWidth
    render_height = $RenderHeight
    conditions = @('persistent', 'forced-cpu-state-roundtrip')
    no_render_allowed = $false
}
$manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $runRoot 'roundtrip_manifest.json') -Encoding UTF8

$runner = Join-Path $ScriptDir 'run_benchmark.ps1'
foreach ($condition in @('persistent', 'forced-cpu-state-roundtrip')) {
    for ($rep = 1; $rep -le $Repetitions; ++$rep) {
        $caseName = if ($condition -eq 'persistent') {
            'persistent-rep{0:D2}' -f $rep
        }
        else {
            'roundtrip-rep{0:D2}' -f $rep
        }
        $caseOutput = Join-Path $runRoot $caseName
        $runArgs = @{
            ProjectRoot = $ProjectRoot
            RunLabel = $RunLabel + '-' + $caseName
            OutputDir = $caseOutput
            SolverVariant = 'gpu-gather-fusion-batched-ls-persistent'
            IterationsPerFrame = $IterationsPerFrame
            Frames = $Frames
            Warmup = $Warmup
            SyncGpu = $true
            DisableVsync = $true
            RenderWidth = $RenderWidth
            RenderHeight = $RenderHeight
            ProcessTimeoutSeconds = $ProcessTimeoutSeconds
            ExtraArgs = @('--scene', $Scene, '--cloth-dimension', $ClothDimension)
        }
        if ($condition -eq 'forced-cpu-state-roundtrip') {
            $runArgs.ForceCpuStateRoundtrip = $true
        }
        Write-Host "[$condition $rep/$Repetitions]"
        & $runner @runArgs
    }
}

if ($PythonExe -eq '') {
    $preferredPython = 'E:\Anaconda\envs\DL\python.exe'
    $PythonExe = if (Test-Path -LiteralPath $preferredPython) { $preferredPython } else { 'python' }
}
& $PythonExe (Join-Path $ScriptDir 'analyze_persistent_roundtrip.py') `
    --run-root $runRoot `
    --expected-frames $Frames `
    --warmup $Warmup `
    --repetitions $Repetitions
if ($LASTEXITCODE -ne 0) {
    throw "Roundtrip analysis failed with exit code $LASTEXITCODE."
}

Write-Host "Roundtrip report: $(Join-Path $runRoot 'roundtrip_report.md')"
Write-Host "Roundtrip figure: $(Join-Path $runRoot 'roundtrip_validation.pdf')"
