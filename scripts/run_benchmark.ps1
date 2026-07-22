param(
    [string]$ProjectRoot = '',
    [string]$RunLabel = ('benchmark-' + (Get-Date -Format 'yyyyMMdd-HHmmss')),
    [int]$Frames = 300,
    [int]$Warmup = 30,
    [string]$OutputDir = '',
    [string]$ExePath = '',
    [switch]$ProfileGpuQueries,
    [switch]$SyncGpu,
    [bool]$NoRender = $true,
    [bool]$Uncapped = $true,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = 'Stop'
$ScriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $PSCommandPath }
if ($ProjectRoot -eq '') {
    $ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $ScriptDir '..'))
}

function Resolve-GenPDExe {
    param(
        [string]$ProjectRoot,
        [string]$RequestedExePath
    )

    if ($RequestedExePath -ne '') {
        return [System.IO.Path]::GetFullPath($RequestedExePath)
    }

    $parent = Split-Path -Parent $ProjectRoot
    $candidates = @(
        (Join-Path $ProjectRoot 'x64\Release\GenPD.exe'),
        (Join-Path $parent 'x64\Release\GenPD.exe')
    )

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return [System.IO.Path]::GetFullPath($candidate)
        }
    }

    throw "GenPD.exe not found. Build Release|x64 first or pass -ExePath."
}

function Set-RunMetadataEnv {
    param([string]$ProjectRoot)

    $git = Get-Command git -ErrorAction SilentlyContinue
    if ($git) {
        $commit = & git -C $ProjectRoot rev-parse --short HEAD 2>$null
        if ($LASTEXITCODE -eq 0 -and $commit) {
            $env:GENPD_GIT_COMMIT = ($commit | Select-Object -First 1).Trim()
        }
    }

    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($nvidiaSmi) {
        $driver = & nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>$null | Select-Object -First 1
        if ($LASTEXITCODE -eq 0 -and $driver) {
            $env:GENPD_NVIDIA_DRIVER_VERSION = $driver.Trim()
        }

        $gpu = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1
        if ($LASTEXITCODE -eq 0 -and $gpu) {
            $env:GENPD_GPU_NAME = $gpu.Trim()
        }
    }
}

$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
if ($OutputDir -eq '') {
    $OutputDir = Join-Path $ProjectRoot (Join-Path 'results' $RunLabel)
}
$OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
$ExePath = Resolve-GenPDExe -ProjectRoot $ProjectRoot -RequestedExePath $ExePath

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
Set-RunMetadataEnv -ProjectRoot $ProjectRoot

$appArgs = @(
    '--benchmark',
    '--project-root', $ProjectRoot,
    '--output-dir', $OutputDir,
    '--run-label', $RunLabel,
    '--frames', $Frames,
    '--warmup', $Warmup
)

if ($NoRender) { $appArgs += '--no-render' }
if ($Uncapped) { $appArgs += '--uncapped' }
if ($SyncGpu) { $appArgs += '--sync-gpu' }
if ($ProfileGpuQueries) { $appArgs += '--profile-gpu-queries' }
if ($ExtraArgs) { $appArgs += $ExtraArgs }

$logPath = Join-Path $OutputDir 'benchmark_stdout.log'
Write-Host "Running: $ExePath $($appArgs -join ' ')"
Push-Location (Split-Path -Parent $ExePath)
try {
    & $ExePath @appArgs *>&1 | Tee-Object -FilePath $logPath
    $exitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

if ($exitCode -ne 0) {
    throw "Benchmark failed with exit code $exitCode. Log: $logPath"
}

Write-Host "Profile CSV: $(Join-Path $OutputDir 'frame_profile.csv')"
Write-Host "Run metadata: $(Join-Path $OutputDir 'run_metadata.json')"
