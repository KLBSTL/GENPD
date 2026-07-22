param(
    [string]$ProjectRoot = '',
    [string]$Configuration = 'Release',
    [string]$Platform = 'x64',
    [string]$MSBuildPath = 'D:\vs2022\MSBuild\Current\Bin\MSBuild.exe'
)

$ErrorActionPreference = 'Stop'
$ScriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $PSCommandPath }
if ($ProjectRoot -eq '') {
    $ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $ScriptDir '..'))
}
$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
$ProjectFile = Join-Path $ProjectRoot 'GenPD.vcxproj'

if (!(Test-Path -LiteralPath $ProjectFile)) {
    throw "GenPD.vcxproj not found: $ProjectFile"
}

if (!(Test-Path -LiteralPath $MSBuildPath)) {
    throw "MSBuild not found: $MSBuildPath"
}

& $MSBuildPath $ProjectFile "/p:Configuration=$Configuration" "/p:Platform=$Platform" /m /v:minimal
exit $LASTEXITCODE
