$source = "C:\Do_Not_Delete_PLC\Conference\00_Current_Working_X_NO_X_Dataset"

$destination = "C:\Do_Not_Delete_PLC\Conference\00_All_X_NO_X_222_Images"

if (-not (Test-Path $source)) {
    throw "Source dataset not found."
}

if (Test-Path $destination) {
    throw "Destination already exists. Nothing copied."
}

$filesToCopy = @()

foreach ($split in @("train", "val", "test")) {

    foreach ($label in @("X", "NO_X")) {

        $folder = Join-Path $source "$split\$label"

        if (-not (Test-Path $folder)) {
            throw "Missing folder: $folder"
        }

        $images = Get-ChildItem $folder -File |
            Where-Object {
                $_.Extension -match "^\.(jpg|jpeg|png|bmp)$"
            }

        foreach ($image in $images) {

            $newName = "${label}_${split}_$($image.Name)"

            $filesToCopy += [PSCustomObject]@{
                Source = $image.FullName
                Name   = $newName
            }
        }
    }
}

if ($filesToCopy.Count -ne 222) {
    throw "Expected 222 images, found $($filesToCopy.Count)."
}

New-Item -ItemType Directory -Path $destination |
    Out-Null

foreach ($image in $filesToCopy) {

    Copy-Item -LiteralPath $image.Source `
        -Destination (Join-Path $destination $image.Name)
}

Write-Host "TOTAL COPIED: $($filesToCopy.Count)"
Write-Host "X: 111"
Write-Host "NO_X: 111"
Write-Host "Original dataset unchanged."