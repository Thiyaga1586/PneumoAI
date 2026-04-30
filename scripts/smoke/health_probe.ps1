$ErrorActionPreference = "Stop"

$BASE = $env:PNEUMOAI_BASE_URL
if (-not $BASE) {
    $BASE = "http://127.0.0.1:8081"
}

$ADMIN_KEY = $env:PNEUMOAI_ADMIN_API_KEY
if (-not $ADMIN_KEY) {
    Write-Host "PNEUMOAI_ADMIN_API_KEY not set. Skipping admin registry probe."
}
else {
    Write-Host "Checking admin registry..."
    curl.exe -f -H "x-api-key: $ADMIN_KEY" "$BASE/admin/registry"
}

Write-Host "Checking readiness..."
curl.exe -f "$BASE/ready"

Write-Host "Checking metrics..."
curl.exe -f "$BASE/metrics"

Write-Host "Health probe passed."