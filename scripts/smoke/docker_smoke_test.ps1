$ErrorActionPreference = "Stop"

$BASE = "http://127.0.0.1:8080"
$IMG = "C:\Users\Thiyaga\OneDrive - SSN-Institute\Projects\Ai and Ml projects\Pneumonia Detection\Data\Final_dataset_cleaned\val\PNEUMONIA\VIRUS-9361672-0002.jpg"

function Wait-Ready {
    param([int]$Retries = 45)

    for ($i = 1; $i -le $Retries; $i++) {
        try {
            $json = curl.exe -s "$BASE/ready"
            if ($LASTEXITCODE -eq 0 -and $json) {
                $ready = $json | ConvertFrom-Json
                if ($ready.status -eq "ready" -and $ready.model_version -eq "v2") {
                    return $ready
                }
            }
        } catch {
            Write-Host "Waiting for API readiness... attempt $i/$Retries"
        }

        Start-Sleep -Seconds 1
    }

    throw "Service did not become ready after $Retries seconds"
}

Write-Host "Checking readiness..."
$ready = Wait-Ready
Write-Host "Ready: model_version=$($ready.model_version), backend=$($ready.backend), device=$($ready.device)"

Write-Host "Checking sync prediction..."
$sync = curl.exe -s -X POST "$BASE/predict-sync" -F "file=@$IMG" | ConvertFrom-Json
if ($sync.status -ne "completed") { throw "Sync prediction failed" }
if ($sync.model_version -ne "v2") { throw "Sync did not serve v2" }
if ($sync.latency_ms -gt 500) { throw "Sync latency too high: $($sync.latency_ms) ms" }

Write-Host "Checking async prediction..."
$queued = curl.exe -s -X POST "$BASE/predict" -F "file=@$IMG" | ConvertFrom-Json
if ($queued.status -ne "queued") { throw "Async enqueue failed" }

$result = $null
for ($i = 1; $i -le 20; $i++) {
    try {
        $result = curl.exe -s "$BASE/predict/$($queued.request_id)" | ConvertFrom-Json
        if ($result.status -eq "completed") { break }
    } catch {}

    Start-Sleep -Seconds 1
}

if ($null -eq $result) { throw "Async status result missing" }
if ($result.status -ne "completed") { throw "Async job did not complete. Status=$($result.status)" }
if ($result.model_version -ne "v2") { throw "Async did not serve v2" }
if ($result.latency_ms -gt 500) { throw "Async latency too high: $($result.latency_ms) ms" }

Write-Host "Docker smoke test passed."