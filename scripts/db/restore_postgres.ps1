param(
    [Parameter(Mandatory=$true)]
    [string]$BackupFile
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $BackupFile)) {
    throw "Backup file not found: $BackupFile"
}

Write-Host "Restoring Postgres backup from: $BackupFile"

Get-Content $BackupFile | docker exec -i pneumoai-postgres psql -U pneumoai -d pneumoai

Write-Host "Restore complete."