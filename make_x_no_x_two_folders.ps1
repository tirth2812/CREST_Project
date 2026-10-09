
$ErrorActionPreference = "Stop"

$source = "C:\Do_Not_Delete_PLC\Conference\00_Current_Working_X_NO_X_Dataset"

$destination = "C:\Do_Not_Delete_PLC\Conference\00_All_X_NO_X_222_By_Class"

if (-not (Test-Path $source)) {
    throw "Source dataset not found."
}

if (Test-Path $destination) {
    throw "Destination already exists. Nothing changed."
}

$expected = @{
    train = 84
    val   = 15
    test  = 12
}

$copyPlan = @()

foreach ($label in @("X", "NO_X")) {

    foreach ($split in @("train", "val", "test")) {

        $folder = Join-Path $source "$split\$label"

        if (-not (Test-Path $folder)) {
            throw "Missing folder: $folder"
        }

        $images = @(
            Get-ChildItem $folder -File |
            Where-Object {
                $_.Extension -match "^\.(jpg|jpeg|png|bmp)$"
            }
        )

        if ($images.Count -ne $expected[$split]) {
            throw "Unexpected count in $folder"
        }

        foreach ($image in $images) {

            $newName = "${split}_$($image.Name)"

            $copyPlan += [PSCustomObject]@{
                Source = $image.FullName
                Label  = $label
                Name   = $newName
            }
        }
    }
}

if ($copyPlan.Count -ne 222) {
    throw "Expected 222 images."
}

New-Item -ItemType Directory `
    -Path (Join-Path $destination "X") `
    -Force | Out-Null

New-Item -ItemType Directory `
    -Path (Join-Path $destination "NO_X") `
    -Force | Out-Null

foreach ($item in $copyPlan) {

    $target = Join-Path `
        (Join-Path $destination $item.Label) `
        $item.Name

    Copy-Item -LiteralPath $item.Source `
        -Destination $target
}

foreach ($label in @("X", "NO_X")) {

    $count = @(
        Get-ChildItem (Join-Path $destination $label) -File
    ).Count

    Write-Host "${label}: $count images"
}

Write-Host "TOTAL: $($copyPlan.Count) images"
Write-Host "Original dataset unchanged."
