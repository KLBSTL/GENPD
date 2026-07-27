param(
    [ValidateSet('hanging', 'moving-sphere')]
    [string] $Scene = 'hanging',
    [int] $Width = 64,
    [int] $Height = 64,
    [int] $Warmup = 8,
    [int] $Frames = 30,
    [int] $SolverIterations = 10,
    [float] $Dt = 0.001,
    [int] $RenderWidth = 1600,
    [int] $RenderHeight = 900,
    [string] $CaptureOutput = '',
    [string] $PeridynoRoot = (Join-Path $PSScriptRoot '..\..\external\peridyno-ppm'),
    [string] $BuildDir = '',
    [string] $OutputDir = ''
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$adapterDir = Join-Path $projectRoot 'external_baselines\peridyno_ppm'
$peridynoPatch = Join-Path $adapterDir 'peridyno-ppm-genpd-operating-point.patch'
if ([string]::IsNullOrWhiteSpace($BuildDir)) {
    $BuildDir = Join-Path $PeridynoRoot 'build-genpd-ppm-cmake331'
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $projectRoot ("results\external-ppm-{0}-{1}" -f (Get-Date -Format 'yyyyMMdd-HHmmss'), $Scene)
}

if (!(Test-Path (Join-Path $PeridynoRoot 'CMakeLists.txt'))) {
    throw "PeriDyno source not found at $PeridynoRoot. Clone the official repository first."
}
if (!(Test-Path $adapterDir)) {
    throw "Tracked PPM adapter source not found at $adapterDir."
}
if (!(Test-Path $peridynoPatch)) {
    throw "Tracked PeriDyno compatibility patch not found at $peridynoPatch."
}

# The adapter deliberately disables self contact, matching GenPD's stated scope. The
# upstream solver otherwise dereferences a self-contact-only buffer in this mode.
$previousErrorAction = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
$applyCheck = & git -c "safe.directory=$PeridynoRoot" -C $PeridynoRoot apply --check --ignore-space-change --whitespace=nowarn $peridynoPatch 2>$null
$applyExitCode = $LASTEXITCODE
if ($applyExitCode -eq 0) {
    & git -c "safe.directory=$PeridynoRoot" -C $PeridynoRoot apply --ignore-space-change --whitespace=nowarn $peridynoPatch
    $ErrorActionPreference = $previousErrorAction
    if ($LASTEXITCODE -ne 0) {
        throw 'Failed to apply the tracked PeriDyno PPM compatibility patch.'
    }
    Write-Host 'Applied tracked PeriDyno PPM compatibility patch.'
}
else {
    $reverseCheck = & git -c "safe.directory=$PeridynoRoot" -C $PeridynoRoot apply --reverse --check --ignore-space-change --whitespace=nowarn $peridynoPatch 2>$null
    $reverseExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorAction
    if ($reverseExitCode -ne 0) {
        throw "PeriDyno source does not match the supported revision or a partial patch is present. Apply exit code: $applyExitCode; reverse exit code: $reverseExitCode"
    }
    Write-Host 'Tracked PeriDyno PPM compatibility patch is already applied.'
}

$exampleDir = Join-Path $PeridynoRoot 'examples\Cuda\CodimensionalPD\GenPD_OperatingPoint'
New-Item -ItemType Directory -Force -Path $exampleDir | Out-Null
Copy-Item -Force (Join-Path $adapterDir 'CMakeLists.txt') (Join-Path $exampleDir 'CMakeLists.txt')
Copy-Item -Force (Join-Path $adapterDir 'GenPD_PPM_OperatingPoint.cpp') (Join-Path $exampleDir 'GenPD_PPM_OperatingPoint.cpp')

$cmake = 'D:\vs2022\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
if (!(Test-Path $cmake)) {
    $cmake = 'cmake.exe'
}

& $cmake -S $PeridynoRoot -B $BuildDir
if ($LASTEXITCODE -ne 0) {
    throw 'PeriDyno CMake configure failed.'
}

& $cmake --build $BuildDir --config Release --target GenPD_PPM_OperatingPoint --parallel 8
if ($LASTEXITCODE -ne 0) {
    throw 'PeriDyno PPM adapter build failed.'
}

$peridynoCommit = (git -c "safe.directory=$PeridynoRoot" -C $PeridynoRoot rev-parse HEAD).Trim()
$executable = Join-Path $BuildDir 'bin\Release\GenPD_PPM_OperatingPoint_Rendered.exe'
if (!(Test-Path $executable)) {
    throw "PPM adapter executable was not produced: $executable"
}

if (![string]::IsNullOrWhiteSpace($CaptureOutput)) {
    & $executable `
        --scene $Scene `
        --width $Width `
        --height $Height `
        --warmup $Warmup `
        --frames $Frames `
        --solver-iterations $SolverIterations `
        --dt $Dt `
        --render-width $RenderWidth `
        --render-height $RenderHeight `
        --capture-output $CaptureOutput `
        --output-dir $OutputDir `
        --peridyno-commit $peridynoCommit
}
else {
    & $executable `
        --scene $Scene `
        --width $Width `
        --height $Height `
        --warmup $Warmup `
        --frames $Frames `
        --solver-iterations $SolverIterations `
        --dt $Dt `
        --render-width $RenderWidth `
        --render-height $RenderHeight `
        --output-dir $OutputDir `
        --peridyno-commit $peridynoCommit
}
if ($LASTEXITCODE -ne 0) {
    throw 'PeriDyno PPM operating-point run failed.'
}

Write-Host "PPM operating-point records: $OutputDir"
