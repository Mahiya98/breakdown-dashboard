$ErrorActionPreference = 'Stop'
$dir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $dir

Write-Host 'Refreshing data from Google Sheets...' -ForegroundColor Cyan
node build.mjs

Write-Host ''
Write-Host 'Starting dashboard server...' -ForegroundColor Cyan
Start-Process 'http://localhost:8090/' | Out-Null
node server.mjs
