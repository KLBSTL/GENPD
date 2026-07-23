$ErrorActionPreference = 'Stop'
$scriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $PSCommandPath }
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptDir '..'))

function Require-Text {
    param([string]$Path, [string]$Pattern)
    if (-not (Select-String -LiteralPath $Path -Pattern $Pattern -Quiet)) {
        throw "Missing diagnostic contract '$Pattern' in $Path"
    }
}

Require-Text (Join-Path $projectRoot 'source\main.cpp') '--verify-cs-gradient'
Require-Text (Join-Path $projectRoot 'source\main.cpp') '--batched-ls-k'
Require-Text (Join-Path $projectRoot 'source\main.cpp') '--armijo-beta'
Require-Text (Join-Path $projectRoot 'source\main.cpp') '--ncg-restart-mode'
Require-Text (Join-Path $projectRoot 'source\main.cpp') '--cloth-width'
Require-Text (Join-Path $projectRoot 'source\simulation.cpp') 'frame_profile_extended.csv'
Require-Text (Join-Path $projectRoot 'source\simulation.cpp') 'validateCSAdjacency'
Require-Text (Join-Path $projectRoot 'source\simulation.cpp') 'm_last_profile_termination_reason'
Require-Text (Join-Path $projectRoot 'source\simulation.cpp') 'm_last_profile_ncg_restarts'
Require-Text (Join-Path $projectRoot 'source\simulation.cpp') 'converge = false;'
Require-Text (Join-Path $projectRoot 'source\simulation.cpp') 'csStateStatsID'
Require-Text (Join-Path $projectRoot 'source\simulation.cpp') 'persistent_collision_dispatches'
Require-Text (Join-Path $projectRoot 'source\simulation.cpp') 'm_cs_gpu_state_valid'
Require-Text (Join-Path $projectRoot 'shaders\descent.comp') 'new_gradient_dot_descent >= 0.0'
Require-Text (Join-Path $projectRoot 'shaders\descent.comp') '_pad0 == 1'
Require-Text (Join-Path $projectRoot 'source\runtime_paths.cpp') 'solver_controls'
Require-Text (Join-Path $projectRoot 'shaders\choose_final.comp') 'step = 0.0;'
Require-Text (Join-Path $projectRoot 'scripts\audit_historical_rendered_results.ps1') 'E0: frame-0 explosion'

Write-Host 'Gradient diagnostic and invalid-state contract passed.'
