param(
    [string]$ProjectRoot = '',
    [string]$ReleaseDir = '',
    [switch]$Overwrite
)

$ErrorActionPreference = 'Stop'

$scriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $PSCommandPath }
if ($ProjectRoot -eq '') {
    $ProjectRoot = Join-Path $scriptDir '..'
}
$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
if ($ReleaseDir -eq '') {
    $ReleaseDir = Join-Path (Split-Path -Parent $ProjectRoot) 'release'
}
$ReleaseDir = [System.IO.Path]::GetFullPath($ReleaseDir)

if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot 'GenPD.vcxproj'))) {
    throw "GenPD project root was not found: $ProjectRoot"
}
if ($ReleaseDir.TrimEnd('\\') -eq $ProjectRoot.TrimEnd('\\') -or $ReleaseDir.Length -lt 4) {
    throw "Refusing to package into an unsafe release directory: $ReleaseDir"
}

$executablePath = Join-Path $ProjectRoot 'x64\Release\GenPD.exe'
if (-not (Test-Path -LiteralPath $executablePath)) {
    throw "Release executable was not found. Build Release|x64 first: $executablePath"
}

if (Test-Path -LiteralPath $ReleaseDir) {
    $existingItems = @(Get-ChildItem -LiteralPath $ReleaseDir -Force)
    if ($existingItems.Count -gt 0) {
        if (-not $Overwrite) {
            throw "Release directory is not empty: $ReleaseDir. Re-run with -Overwrite to replace this package."
        }
        Remove-Item -LiteralPath $ReleaseDir -Recurse -Force
    }
}
New-Item -ItemType Directory -Path $ReleaseDir -Force | Out-Null

$assetDirectories = @('config', 'shaders', 'scenes', 'textures', 'mesh_models', 'obj_models')
foreach ($assetDirectory in $assetDirectories) {
    $source = Join-Path $ProjectRoot $assetDirectory
    if (-not (Test-Path -LiteralPath $source)) {
        throw "Required runtime asset directory is missing: $source"
    }
    $destination = Join-Path $ReleaseDir $assetDirectory
    Copy-Item -LiteralPath $source -Destination $destination -Recurse -Force
}

$obsoleteConfigFiles = @(
    Join-Path $ReleaseDir 'config\handle.txt.81grid-backup-20260604'
)
foreach ($obsoleteConfigFile in $obsoleteConfigFiles) {
    if (Test-Path -LiteralPath $obsoleteConfigFile) {
        Remove-Item -LiteralPath $obsoleteConfigFile -Force
    }
}

Copy-Item -LiteralPath $executablePath -Destination (Join-Path $ReleaseDir 'GenPD.exe') -Force

$applicationDlls = @('AntTweakBar64.dll', 'freeglut.dll', 'glew32.dll')
foreach ($dllName in $applicationDlls) {
    $sourceCandidates = @(
        (Join-Path -Path $ProjectRoot -ChildPath $dllName)
        (Join-Path -Path (Join-Path -Path $ProjectRoot -ChildPath 'Release') -ChildPath $dllName)
    )
    $source = $sourceCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $source) {
        throw "Required application DLL was not found: $dllName"
    }
    Copy-Item -LiteralPath $source -Destination (Join-Path $ReleaseDir $dllName) -Force
}

$redistBase = 'D:\vs2022\VC\Redist\MSVC'
if (-not (Test-Path -LiteralPath $redistBase)) {
    throw "Visual C++ Redistributable directory was not found: $redistBase"
}
$redistVersion = Get-ChildItem -LiteralPath $redistBase -Directory |
    Where-Object { $_.Name -match '^\d' } |
    Sort-Object { [version]$_.Name } -Descending |
    Select-Object -First 1
if (-not $redistVersion) {
    throw "No versioned Visual C++ Redistributable directory was found under: $redistBase"
}

$runtimeDllSources = @(
    (Join-Path -Path $redistVersion.FullName -ChildPath 'x64\Microsoft.VC143.CRT\msvcp140.dll')
    (Join-Path -Path $redistVersion.FullName -ChildPath 'x64\Microsoft.VC143.CRT\vcruntime140.dll')
    (Join-Path -Path $redistVersion.FullName -ChildPath 'x64\Microsoft.VC143.CRT\vcruntime140_1.dll')
    (Join-Path -Path $redistVersion.FullName -ChildPath 'x64\Microsoft.VC143.OpenMP\vcomp140.dll')
)
foreach ($source in $runtimeDllSources) {
    if (-not (Test-Path -LiteralPath $source)) {
        throw "Required Visual C++ runtime DLL was not found: $source"
    }
    Copy-Item -LiteralPath $source -Destination (Join-Path $ReleaseDir (Split-Path -Leaf $source)) -Force
}

New-Item -ItemType Directory -Path (Join-Path $ReleaseDir 'results') -Force | Out-Null

$readme = @(
    'GenPD release package',
    '',
    'Run GenPD.exe directly from this folder. Keep the executable, DLLs, and asset directories together.',
    'The program writes benchmark output to results\\<run-label> by default.',
    '',
    'Example:',
    '  .\\GenPD.exe --benchmark --frames 300 --warmup 30 --uncapped --run-label release-benchmark',
    '',
    'Requirements: Windows x64, a working OpenGL driver, and a GPU supported by the application.'
)
Set-Content -LiteralPath (Join-Path $ReleaseDir 'README.txt') -Value $readme -Encoding ascii

$manifest = [ordered]@{
    package_format = 'GenPD-runtime-package-v1'
    created_utc = (Get-Date).ToUniversalTime().ToString('o')
    source_executable = $executablePath
    source_executable_sha256 = (Get-FileHash -LiteralPath $executablePath -Algorithm SHA256).Hash
    visual_cpp_redist = $redistVersion.FullName
    required_runtime_inputs = @('config', 'shaders', 'scenes', 'textures', 'mesh_models', 'obj_models')
    bundled_dlls = @($applicationDlls + ($runtimeDllSources | ForEach-Object { Split-Path -Leaf $_ }))
}
$manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $ReleaseDir 'release_manifest.json') -Encoding ascii

Write-Output "Packaged GenPD release: $ReleaseDir"
Write-Output "Executable: $(Join-Path $ReleaseDir 'GenPD.exe')"
