# Breakdown Analysis Dashboard

Interactive dashboard that pulls machine **breakdown / downtime** logs from Google Sheets,
normalises them, and shows charts + KPIs with filters for **SBU, Mill/Machine, Breakdown Type,
Year and Date range**.

## Run

Double-click `start.bat` (or run `.\start.ps1`). It will:

1. `node build.mjs` – download all sheets and (re)generate `data\data.json` + `data\meta.json`
2. open `http://localhost:8090/` and serve the dashboard

Open: **http://localhost:8090/**

## Files

| File | Purpose |
|------|---------|
| `build.mjs` | Fetches the Google Sheets, parses/normalises rows, writes `data\data.json` |
| `dashboard.template.html` | Dashboard UI (filters + Chart.js charts). Source of `dashboard.html` |
| `dashboard.html` | Generated page (loads `data/data.json` at runtime) |
| `server.mjs` | Tiny static file server on port 8090 |
| `vendor/chart.umd.min.js` | Chart.js (bundled locally, works offline) |
| `data/` | Downloaded CSVs + parsed `data.json` / `meta.json` |

## Data sources

13 Google Sheets (tabs named "<x> Breakdown"). Two are **not shared publicly** and are skipped
automatically (a note is shown in the dashboard):

- Bending Breakdown
- Tapeline Breakdown

To include them: set link sharing to *Anyone with the link → Viewer*, then re-run `node build.mjs`.

## Adding / changing sheets

Edit the `SHEETS` array at the top of `build.mjs` (name, spreadsheet id, gid, fallback SBU).
Column names are auto-detected (SBU/section, Date, Mill/Machine, Breakdown Type, Breakdown Name,
Reason, Downtime, Shift, Month, …), so slight header differences across sheets are handled.
