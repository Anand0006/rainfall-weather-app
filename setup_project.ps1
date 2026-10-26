$p1   = "C:\p1"
$raw  = "$p1\raw"
$proc = "$p1\processed"
$out  = "$p1\output"
$be   = "$p1\backend"

New-Item -ItemType Directory -Force -Path $raw, $proc, $be, "$p1\Temperature" | Out-Null

# Raw inputs go to raw\
foreach ($f in @("district_monthly_rainfall_1901_2025.csv",
                 "gadm41_IND_2.json",
                 "Global_Landslide_Catalog_Export_rows.csv")) {
    if (Test-Path "$p1\$f") { Move-Item "$p1\$f" $raw -Force }
}

# Processed summaries sitting at the top level go to processed\
foreach ($f in @("district_data_quality.csv", "dry_region_metrics.csv")) {
    if (Test-Path "$p1\$f") { Move-Item "$p1\$f" $proc -Force }
}

# Everything in output\ goes to processed\, then output\ is removed if empty
if (Test-Path $out) {
    Get-ChildItem $out | Move-Item -Destination $proc -Force
    if (-not (Get-ChildItem $out)) { Remove-Item $out }
}

Write-Host "`nraw:" -ForegroundColor Cyan
Get-ChildItem $raw | Select-Object Name, Length | Format-Table -AutoSize
Write-Host "processed:" -ForegroundColor Cyan
Get-ChildItem $proc | Select-Object Name, Length | Format-Table -AutoSize