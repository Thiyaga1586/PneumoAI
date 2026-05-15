$ErrorActionPreference = "Stop"

$BackupDir = "data/backups/postgres"
New-Item -ItemType Directory -Force $BackupDir | Out-Null

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$BackupFile = "$BackupDir/pneumoai_$Timestamp.sql"

Write-Host "Creating Postgres backup: $BackupFile"

docker exec pneumoai-postgres pg_dump -U pneumoai -d pneumoai > $BackupFile

Write-Host "Backup complete: $BackupFile"