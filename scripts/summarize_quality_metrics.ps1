param(
    [Parameter(Mandatory = $true)]
    [string]$InputCsv,
    [string]$OutputCsv = ''
)

$ErrorActionPreference = 'Stop'

function Convert-QualityDouble {
    param([object]$Value)

    $parsed = 0.0
    if ([double]::TryParse(
        [string]$Value,
        [Globalization.NumberStyles]::Float,
        [Globalization.CultureInfo]::InvariantCulture,
        [ref]$parsed)) {
        return $parsed
    }
    return $null
}

function Get-Quantile {
    param(
        [double[]]$Values,
        [double]$Quantile
    )

    if ($Values.Count -eq 0) {
        return $null
    }

    $sorted = @($Values | Sort-Object)
    $index = ($sorted.Count - 1) * $Quantile
    $lower = [math]::Floor($index)
    $upper = [math]::Ceiling($index)
    if ($lower -eq $upper) {
        return $sorted[$lower]
    }

    return $sorted[$lower] + ($sorted[$upper] - $sorted[$lower]) * ($index - $lower)
}

function Add-MetricSummary {
    param(
        [System.Collections.Generic.List[object]]$Rows,
        [string]$Metric,
        [object[]]$Values
    )

    $numbers = [System.Collections.Generic.List[double]]::new()
    foreach ($value in $Values) {
        $parsed = Convert-QualityDouble $value
        if ($null -ne $parsed -and -not [double]::IsNaN($parsed) -and -not [double]::IsInfinity($parsed)) {
            $numbers.Add($parsed)
        }
    }

    if ($numbers.Count -eq 0) {
        return
    }

    $mean = ($numbers | Measure-Object -Average).Average
    $variance = 0.0
    foreach ($value in $numbers) {
        $variance += ($value - $mean) * ($value - $mean)
    }
    $stddev = [math]::Sqrt($variance / $numbers.Count)
    $sorted = @($numbers | Sort-Object)
    $Rows.Add([pscustomobject]@{
        metric = $Metric
        count = $numbers.Count
        mean = $mean
        stddev = $stddev
        p50 = Get-Quantile -Values $sorted -Quantile 0.5
        p95 = Get-Quantile -Values $sorted -Quantile 0.95
        minimum = $sorted[0]
        maximum = $sorted[$sorted.Count - 1]
    })
}

$InputCsv = [IO.Path]::GetFullPath($InputCsv)
if ($OutputCsv -eq '') {
    $OutputCsv = Join-Path (Split-Path -Parent $InputCsv) 'quality_summary.csv'
}
$OutputCsv = [IO.Path]::GetFullPath($OutputCsv)

if (-not (Test-Path -LiteralPath $InputCsv)) {
    throw "Input CSV was not found: $InputCsv"
}

$qualityRows = @(Import-Csv -LiteralPath $InputCsv)
if ($qualityRows.Count -eq 0) {
    throw "Input CSV has no data rows: $InputCsv"
}

$referenceRows = @($qualityRows | Where-Object { $_.has_reference -eq '1' })
$summaryRows = [System.Collections.Generic.List[object]]::new()

@('position_rel_l2', 'velocity_rel_l2', 'constraint_energy_rel_error') | ForEach-Object {
    $metric = $_
    if ($referenceRows.Count -gt 0 -and $null -ne $referenceRows[0].PSObject.Properties[$metric]) {
        Add-MetricSummary -Rows $summaryRows -Metric $metric -Values @($referenceRows | ForEach-Object { $_.$metric })
    }
}

@('constraint_energy', 'mean_stretch_strain', 'max_stretch_strain', 'max_penetration_depth') | ForEach-Object {
    $metric = $_
    if ($null -ne $qualityRows[0].PSObject.Properties[$metric]) {
        Add-MetricSummary -Rows $summaryRows -Metric $metric -Values @($qualityRows | ForEach-Object { $_.$metric })
    }
}

$failureValues = @($qualityRows | ForEach-Object {
    if ($_.finite -ne '1' -or $_.exploded -ne '0') { 1.0 } else { 0.0 }
})
Add-MetricSummary -Rows $summaryRows -Metric 'failure_rate' -Values $failureValues

$outputDirectory = Split-Path -Parent $OutputCsv
if ($outputDirectory -ne '') {
    New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
}
$summaryRows | Export-Csv -LiteralPath $OutputCsv -NoTypeInformation
Write-Output "Quality summary: $OutputCsv"
