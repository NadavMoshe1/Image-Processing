# Zip one YOLO dataset folder for upload to Google Drive.
# Usage: .\scripts\zip_yolo_for_colab.ps1 noise_snr_10db
#        .\scripts\zip_yolo_for_colab.ps1 all

param(
    [Parameter(Position = 0)]
    [ValidateSet("noise_snr_10db", "low_light_gamma_0.35", "jpeg_q20", "all")]
    [string]$Dataset = "all"
)

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$SrcRoot = Join-Path $Root "data\yolo_distorted"
$OutDir = Join-Path $Root "colab_upload"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$Names = if ($Dataset -eq "all") {
    @("noise_snr_10db", "low_light_gamma_0.35", "jpeg_q20")
} else {
    @($Dataset)
}

foreach ($name in $Names) {
    $folder = Join-Path $SrcRoot $name
    if (-not (Test-Path $folder)) {
        Write-Warning "Skip missing: $folder"
        continue
    }
    $zip = Join-Path $OutDir "$name.zip"
    if (Test-Path $zip) { Remove-Item $zip -Force }
    Compress-Archive -Path $folder -DestinationPath $zip -CompressionLevel Optimal
    $mb = [math]::Round((Get-Item $zip).Length / 1MB, 1)
    Write-Host "Created $zip ($mb MB)"
}

Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Upload zip(s) from colab_upload\ to Google Drive (browser upload is resumable)"
Write-Host "  2. In Colab notebook: mount Drive and unzip (see notebooks/colab_finetune.ipynb)"
