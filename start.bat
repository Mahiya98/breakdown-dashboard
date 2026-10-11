@echo off
cd /d "%~dp0"
echo Refreshing data from Google Sheets...
node build.mjs
echo.
echo Starting dashboard at http://localhost:8090/
start "" http://localhost:8090/
node server.mjs
