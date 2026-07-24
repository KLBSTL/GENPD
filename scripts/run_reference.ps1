param(
    [string]$ProjectRoot = '',
    [string]$RunLabel = ('reference-' + (Get-Date -Format 'yyyyMMdd-HHmmss')),
    [int]$Frames = 300,
    [int]$Warmup = 30,
    [int]$ReferenceIterations = 100,
    [int]$CheckpointStride = 10,
    [string]$OutputDir = '',
    [string]$ExePath = '',
    [switch]$ProfileGpuQueries,
    [switch]$SyncGpu,
    [switch]$DisableVsync,
    [int]$RenderWidth = 0,
    [int]$RenderHeight = 0,
    [int]$ProcessTimeoutSeconds = 0,
    [bool]$NoRender = $false,
    [bool]$Uncapped = $true,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = 'Stop'
$ScriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $PSCommandPath }
if ($ProjectRoot -eq '') {
    $ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $ScriptDir '..'))
}
if ($ReferenceIterations -lt 1) {
    throw 'ReferenceIterations must be positive.'
}
if ($CheckpointStride -lt 1) {
    throw 'CheckpointStride must be positive.'
}

$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
if ($OutputDir -eq '') {
    $OutputDir = Join-Path $ProjectRoot (Join-Path 'results' $RunLabel)
}
$OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
$checkpointDir = Join-Path $OutputDir 'reference_checkpoints'
$benchmarkScript = Join-Path $ScriptDir 'run_benchmark.ps1'

& $benchmarkScript -ProjectRoot $ProjectRoot -RunLabel $RunLabel -Frames $Frames -Warmup $Warmup `
    -SolverVariant cpu-ncg -IterationsPerFrame $ReferenceIterations `
    -ReferenceExportDir $checkpointDir -QualityCheckpointStride $CheckpointStride `
    -OutputDir $OutputDir -ExePath $ExePath -ProfileGpuQueries:$ProfileGpuQueries `
    -SyncGpu:$SyncGpu -DisableVsync:$DisableVsync -RenderWidth $RenderWidth -RenderHeight $RenderHeight `
    -ProcessTimeoutSeconds $ProcessTimeoutSeconds `
    -NoRender:$NoRender -Uncapped:$Uncapped -ExtraArgs $ExtraArgs
if (-not $?) {
    throw 'Reference benchmark failed.'
}

Write-Host "Reference checkpoints: $checkpointDir"
