param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'
$scriptPath = Join-Path $ProjectRoot 'scripts\package_release.ps1'
if (-not (Test-Path -LiteralPath $scriptPath)) {
    throw "Release packaging script is missing: $scriptPath"
}

$scriptText = Get-Content -LiteralPath $scriptPath -Raw
$requiredTokens = @(
    "'config', 'shaders', 'scenes', 'textures', 'mesh_models', 'obj_models'",
    "'AntTweakBar64.dll', 'freeglut.dll', 'glew32.dll'",
    'Microsoft.VC143.CRT\msvcp140.dll',
    'Microsoft.VC143.CRT\vcruntime140.dll',
    'Microsoft.VC143.CRT\vcruntime140_1.dll',
    'Microsoft.VC143.OpenMP\vcomp140.dll',
    "'release_manifest.json'",
    "'README.txt'"
)
foreach ($token in $requiredTokens) {
    if ($scriptText.IndexOf($token, [System.StringComparison]::Ordinal) -lt 0) {
        throw "Release packaging contract is missing: $token"
    }
}

$runtimePaths = Get-Content -LiteralPath (Join-Path $ProjectRoot 'source\runtime_paths.cpp') -Raw
if ($runtimePaths.IndexOf('source\\main.cpp', [System.StringComparison]::Ordinal) -ge 0) {
    throw 'Runtime root detection still requires source\\main.cpp, so a deployed release package cannot self-resolve.'
}

Write-Output 'Release packaging contract passed.'
