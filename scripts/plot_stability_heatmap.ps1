param(
    [Parameter(Mandatory = $true)]
    [string]$InputCsv,
    [string]$OutputSvg = ''
)

$ErrorActionPreference = 'Stop'
$invariant = [System.Globalization.CultureInfo]::InvariantCulture
$InputCsv = [System.IO.Path]::GetFullPath($InputCsv)
if (-not (Test-Path -LiteralPath $InputCsv)) { throw "Stability CSV was not found: $InputCsv" }
if ($OutputSvg -eq '') {
    $OutputSvg = Join-Path (Split-Path -Parent $InputCsv) 'stability_heatmap.svg'
}
$OutputSvg = [System.IO.Path]::GetFullPath($OutputSvg)

$rows = @(Import-Csv -LiteralPath $InputCsv)
if ($rows.Count -eq 0) { throw 'Stability CSV is empty.' }
foreach ($column in @('timestep', 'stretch_stiffness', 'stable', 'failure_rate')) {
    if ($rows[0].PSObject.Properties.Name -notcontains $column) {
        throw "Stability CSV is missing required column: $column"
    }
}

function To-Number([string]$value) {
    return [double]::Parse($value, [System.Globalization.NumberStyles]::Float, $invariant)
}
function Escape-Xml([string]$value) {
    return [System.Security.SecurityElement]::Escape($value)
}

$timesteps = @($rows | ForEach-Object { To-Number $_.timestep } | Sort-Object -Unique)
$stiffnesses = @($rows | ForEach-Object { To-Number $_.stretch_stiffness } | Sort-Object -Unique)
$cellWidth = 132
$cellHeight = 64
$left = 145
$top = 70
$width = $left + $cellWidth * $stiffnesses.Count + 24
$height = $top + $cellHeight * $timesteps.Count + 72

$svg = New-Object System.Collections.Generic.List[string]
$svg.Add("<svg xmlns=`"http://www.w3.org/2000/svg`" width=`"$width`" height=`"$height`" viewBox=`"0 0 $width $height`">")
$svg.Add('<rect width="100%" height="100%" fill="#ffffff"/>')
$svg.Add('<style>text{font-family:Arial,sans-serif;fill:#1f2933} .title{font-size:18px;font-weight:bold} .axis{font-size:12px} .cell{font-size:13px;font-weight:bold;fill:#ffffff}</style>')
$svg.Add('<text class="title" x="16" y="28">GenPD stability map: timestep x stretch stiffness</text>')
$svg.Add('<text class="axis" x="16" y="49">green: no finite/explosion failure; red: failure observed</text>')

for ($x = 0; $x -lt $stiffnesses.Count; ++$x) {
    $label = $stiffnesses[$x].ToString('G', $invariant)
    $center = $left + $x * $cellWidth + $cellWidth / 2
    $svg.Add("<text class=`"axis`" text-anchor=`"middle`" x=`"$center`" y=`"$($top - 12)`">$label</text>")
}
$svg.Add("<text class=`"axis`" x=`"$left`" y=`"$($top - 34)`">stretch stiffness</text>")

for ($y = 0; $y -lt $timesteps.Count; ++$y) {
    $timestep = $timesteps[$y]
    $rowY = $top + $y * $cellHeight
    $timestepLabel = $timestep.ToString('G', $invariant)
    $svg.Add("<text class=`"axis`" text-anchor=`"end`" x=`"$($left - 10)`" y=`"$($rowY + 36)`">$timestepLabel</text>")
    for ($x = 0; $x -lt $stiffnesses.Count; ++$x) {
        $stiffness = $stiffnesses[$x]
        $match = @($rows | Where-Object { (To-Number $_.timestep) -eq $timestep -and (To-Number $_.stretch_stiffness) -eq $stiffness } | Select-Object -First 1)
        $cellX = $left + $x * $cellWidth
        $stable = $match.Count -gt 0 -and $match[0].stable -eq '1'
        $fill = if ($stable) { '#2e8b57' } else { '#c53b33' }
        $status = if ($stable) { 'stable' } else { 'failed' }
        $rate = if ($match.Count -gt 0) { ('fail {0:P1}' -f (To-Number $match[0].failure_rate)) } else { 'missing' }
        $svg.Add("<rect x=`"$cellX`" y=`"$rowY`" width=`"$($cellWidth - 4)`" height=`"$($cellHeight - 4)`" fill=`"$fill`"/>" )
        $svg.Add("<text class=`"cell`" text-anchor=`"middle`" x=`"$($cellX + ($cellWidth - 4) / 2)`" y=`"$($rowY + 28)`">$(Escape-Xml $status)</text>")
        $svg.Add("<text class=`"cell`" text-anchor=`"middle`" x=`"$($cellX + ($cellWidth - 4) / 2)`" y=`"$($rowY + 46)`">$(Escape-Xml $rate)</text>")
    }
}
$svg.Add("<text class=`"axis`" x=`"16`" y=`"$($top + 18)`">timestep</text>")
$svg.Add('</svg>')

[System.IO.Directory]::CreateDirectory((Split-Path -Parent $OutputSvg)) | Out-Null
[System.IO.File]::WriteAllLines($OutputSvg, $svg, [System.Text.UTF8Encoding]::new($false))
Write-Host "Stability heatmap: $OutputSvg"
