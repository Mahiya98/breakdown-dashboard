import { writeFileSync, mkdirSync, existsSync, readFileSync } from 'node:fs';

const OUT_DIR = 'E:\\BreakdownDashboard';
const DATA_DIR = OUT_DIR + '\\data';
if (!existsSync(DATA_DIR)) mkdirSync(DATA_DIR, { recursive: true });

// ---------- sheet registry ----------
// sbu = fallback SBU for rows that leave the SBU/section column blank
const SHEETS = [
  { name: 'Rolling Breakdown',         id: '1iFTy9-9w3HI3A6lgP6PDl3kOb0rz-7Nfcwwu6UhQRqo', gid: '482789258',   sbu: 'AIL' },
  { name: 'Bending Breakdown',         id: '190p7ou5_uYHAXTdxGdUmx9Jc_XhlTCJ3zMz-3Geb_HE', gid: '482789258',   sbu: 'AIL' },
  { name: 'SMS Breakdown',             id: '186Nt9r1_7-Ij3s5PmFA3rghdy8sxhUBqHn6g5L3tlTA', gid: '482789258',   sbu: 'AIL' },
  { name: 'Breakdown E1',              id: '1ElfWAADdrvszMVLN5A3Cw4IWnvrsheT_tf1kaGfwREU', gid: '482789258',   sbu: 'MRML' },
  { name: 'Breakdown K1',              id: '1KjDYnm1D0B8D91BhhmZ8Sm1XuRU2gT20mpdwbBdkkDs', gid: '482789258',   sbu: 'FAL' },
  { name: 'VRM Breakdown',             id: '1NzcGrUHk7N4FSVGpAp9ZQ_H7TbpmX92YTtaJSnMx8Hg', gid: '482789258',   sbu: 'ACCL' },
  { name: 'Packer Breakdown',          id: '1A-qefcx2jNDVB14DvuxmXFUl4bJJG8a9NEapCZYGxcA', gid: '482789258',   sbu: 'ACCL' },
  { name: 'Loom U1 Breakdown',         id: '1Q8gbAgtvxw9BaumOzPCwHpbZuzsW0tdNA_LlCpgP1jM', gid: '1119362862',  sbu: 'APFIL Unit-1 Loom' },
  { name: 'Loom U2 Breakdown',         id: '1H5YOsacylfSGxHFGA-D9JD7LCnWSUsx4NtU-fjccZ7w', gid: '482789258',   sbu: 'APFIL Unit-2 Loom' },
  { name: 'Tapeline Breakdown',        id: '1Noxy7HFM-N98NUTjmWji3s364myGGBfQiOWToAtuRns', gid: '482789258',   sbu: 'APFIL' },
  { name: 'Flour Breakdown',           id: '1EUn0YYjbM5CsJL5-O2tQRF5s2C_VcHvqEoo0B_B6hw0', gid: '482789258',   sbu: 'AEFML' },
  { name: 'Breakdown D1',              id: '1sRaACw7K7q4YN8KfkmdEMItgxNQpqzjmWje1TXrUBqg', gid: '482789258',   sbu: 'AEL' },
  { name: 'Daal-Red Lentil Breakdown', id: '1uQCYRIxvWdF4LgYwBJJDRwKKRYcVpjiuYn5LYYd6tno', gid: '482789258',   sbu: 'AEL' },
];

// ---------- CSV parse ----------
function parseCSV(text) {
  const rows = [];
  let row = [], field = '', inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (inQuotes) {
      if (ch === '"') { if (text[i + 1] === '"') { field += '"'; i++; } else inQuotes = false; }
      else field += ch;
    } else {
      if (ch === '"') inQuotes = true;
      else if (ch === ',') { row.push(field); field = ''; }
      else if (ch === '\n') { row.push(field); rows.push(row); row = []; field = ''; }
      else if (ch === '\r') { /* skip */ }
      else field += ch;
    }
  }
  if (field.length > 0 || row.length > 0) { row.push(field); rows.push(row); }
  return rows;
}

const MONTHS = { jan: 1, feb: 2, mar: 3, apr: 4, may: 5, jun: 6, jul: 7, aug: 8, sep: 9, oct: 10, nov: 11, dec: 12 };
const MONTH_NAMES = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const monthNum = (s) => MONTHS[String(s).trim().toLowerCase().slice(0, 3)] || null;

function parseDate(s) {
  if (!s) return null;
  const t = String(s).trim();
  const m = t.match(/^(\d{1,2})[\/\-.](\d{1,2})[\/\-.](\d{2,4})/);
  if (!m) return null;
  let a = +m[1], b = +m[2], y = +m[3]; if (y < 100) y += 2000;
  const md = new Date(y, a - 1, b), dm = new Date(y, b - 1, a);
  const mdOk = md.getFullYear() === y && md.getMonth() === a - 1 && md.getDate() === b;
  const dmOk = dm.getFullYear() === y && dm.getMonth() === b - 1 && dm.getDate() === a;
  if (a > 12 && dmOk) return { y, m: b, d: a };
  if (b > 12 && mdOk) return { y, m: a, d: b };
  if (mdOk) return { y, m: a, d: b };
  if (dmOk) return { y, m: b, d: a };
  return null;
}

function parseMonthLabel(s) {
  // "Nov-24" | "December-2024" | "January-2025" | "9" | "2024" | "Nov-2024"
  if (s == null) return {};
  const t = String(s).trim();
  if (!t) return {};
  const m = t.match(/^([A-Za-z]{3,})[^0-9]*(\d{2,4})?/);
  if (m) { const mon = monthNum(m[1]); let y = m[2] ? +m[2] : null; if (y && y < 100) y += 2000; if (y && (y < 2000 || y > 2035)) y = null; return { month: mon, year: y }; }
  const n = +t;
  if (!isNaN(n)) { if (n >= 1 && n <= 12) return { month: n }; if (n >= 2000 && n <= 2100) return { year: n }; }
  return {};
}

const norm = (h) => String(h || '').toLowerCase().replace(/[^a-z0-9]/g, '');
const clean = (v) => (v == null ? '' : String(v).replace(/\s+/g, ' ').trim());

// Collapse SBU variants (e.g. "AEL (Mustard Oil)", "APFIL Unit-1 Loom") to a base code.
function baseSbu(s, fallback) {
  const t = clean(s);
  const m = t.match(/^[A-Za-z&]+/);
  return m ? m[0].toUpperCase() : fallback;
}
function unitTag(sheetName) { const m = sheetName.match(/U(\d)/i); return m ? 'U' + m[1] : ''; }

function findIdx(head, names) {
  for (const nm of names) { const i = head.indexOf(nm); if (i >= 0) return i; }
  return -1;
}

// canonical column recognition (normalized header -> field)
const FIELDS = {
  sbu:         ['sbu', 'sectionunit', 'unit', 'column1'],
  date:        ['date', 'breakdowndate'],
  shiftName:   ['shiftname'],
  machine:     ['millname', 'mcname', 'section', 'machine', 'machinename'],
  product:     ['productcriteria', 'product'],
  operator:    ['operatorname', 'shiftinchargename', 'inchargename'],
  supervisor:  ['supervisorname'],
  shift:       ['shift'],
  type:        ['breakdowntype', 'category', 'type'],
  breakdownName: ['breakdwonname', 'breakdownname'],
  reason:      ['reason', 'breakdownreason'],
  downtime:    ['downtime'],
  remarks:     ['remarks', 'remark'],
  month:       ['month', 'months', 'monthname'],
  monthNo:     ['monthno'],
};

function buildIndex(headerRow) {
  const head = headerRow.map(norm);
  const idx = {};
  for (const [field, names] of Object.entries(FIELDS)) idx[field] = findIdx(head, names);
  // explicit numeric month column "Month No."
  return idx;
}

function get(row, idx) { return idx >= 0 && idx < row.length ? clean(row[idx]) : ''; }

function num(s) {
  if (s == null) return 0;
  const n = parseFloat(String(s).replace(/[^0-9.\-]/g, ''));
  return isNaN(n) ? 0 : n;
}

// ---------- fetch + normalize ----------
const all = [];
const sheetStats = [];
const errors = [];

for (const sheet of SHEETS) {
  const url = `https://docs.google.com/spreadsheets/d/${sheet.id}/export?format=csv&gid=${sheet.gid}`;
  let csv;
  try {
    const res = await fetch(url, { redirect: 'follow' });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    csv = await res.text();
  } catch (e) {
    errors.push({ sheet: sheet.name, error: e.message });
    console.log(`  ${sheet.name}: SKIPPED (${e.message})`);
    continue;
  }
  try { writeFileSync(`${DATA_DIR}\\${sheet.name}.csv`, csv); } catch {}

  const rows = parseCSV(csv);
  const nonEmpty = rows.filter((r) => r.some((c) => clean(c) !== ''));
  if (!nonEmpty.length) { errors.push({ sheet: sheet.name, error: 'empty' }); continue; }

  const hdrIdx = 0;
  const idx = buildIndex(nonEmpty[hdrIdx]);
  let count = 0;
  let totalDt = 0;

  for (let i = hdrIdx + 1; i < nonEmpty.length; i++) {
    const r = nonEmpty[i];
    const dateRaw = get(r, idx.date);
    const monthRaw = get(r, idx.month);
    const monthNoRaw = get(r, idx.monthNo);
    const dt = num(get(r, idx.downtime));

    const d = parseDate(dateRaw);
    let ml = parseMonthLabel(monthRaw);
    if (!ml.month && monthNoRaw) ml = { month: +monthNoRaw || ml.month, year: ml.year };

    let year = d ? d.y : null;
    let month = d ? d.m : null;
    if (ml.month && (!d || d.m !== ml.month)) month = ml.month;
    if (!year && ml.year) year = ml.year;
    if (!month && ml.month) month = ml.month;

    const rawType = get(r, idx.type);
    const rawMachine = get(r, idx.machine);
    const reason = get(r, idx.reason);
    const bname = get(r, idx.breakdownName);

    // skip rows that have no substance at all (blank template / trailing rows)
    if (!(dateRaw || rawType || reason || bname || dt || rawMachine)) continue;

    const breakdownType = rawType || '(blank)';
    let machine = rawMachine || '(blank)';
    const tag = unitTag(sheet.name);
    if (tag && /^\d+$/.test(rawMachine)) machine = tag + ' #' + rawMachine;

    const sbu = baseSbu(get(r, idx.sbu), baseSbu(sheet.sbu));

    all.push({
      sheet: sheet.name,
      sbu,
      date: dateRaw,
      ts: d ? Date.UTC(d.y, d.m - 1, d.d) : null,
      year,
      month,
      monthName: month ? MONTH_NAMES[month] : '',
      ym: year && month ? year + '-' + String(month).padStart(2, '0') : '',
      shiftName: get(r, idx.shiftName),
      machine,
      product: get(r, idx.product),
      operator: get(r, idx.operator),
      supervisor: get(r, idx.supervisor),
      shift: get(r, idx.shift),
      breakdownType,
      breakdownName: bname,
      reason,
      downtime: dt,
      remarks: get(r, idx.remarks),
    });
    count++; totalDt += dt;
  }

  sheetStats.push({ sheet: sheet.name, rows: count, downtime: totalDt });
  console.log(`  ${sheet.name}: ${count} rows, ${Math.round(totalDt)} min downtime`);
}

all.sort((a, b) => (b.ts || 0) - (a.ts || 0));

// Merge values that differ only by letter-case (e.g. "BULK" vs "Bulk"),
// keeping the most common spelling (tic: prefer fewer capitals, then A-Z).
function canonicalize(field) {
  const groups = new Map(); // lower -> { value: count }
  for (const r of all) {
    const v = r[field];
    if (!v) continue;
    const k = v.toLowerCase();
    if (!groups.has(k)) groups.set(k, new Map());
    const g = groups.get(k);
    g.set(v, (g.get(v) || 0) + 1);
  }
  const canon = new Map();
  const caps = (v) => { let n = 0; for (const c of v) if (c >= 'A' && c <= 'Z') n++; return n; };
  for (const [k, g] of groups) {
    let best = '', bc = -1, bcap = Infinity;
    for (const [v, c] of g) {
      const cp = caps(v);
      if (c > bc || (c === bc && cp < bcap) || (c === bc && cp === bcap && v < best)) { best = v; bc = c; bcap = cp; }
    }
    canon.set(k, best);
  }
  for (const r of all) if (r[field]) r[field] = canon.get(r[field].toLowerCase());
}
['machine', 'breakdownType', 'shift', 'sbu'].forEach(canonicalize);

const sbuList = [...new Set(all.map((r) => r.sbu))].filter(Boolean).sort();
const machineList = [...new Set(all.map((r) => r.machine))].filter(Boolean).sort();
const typeList = [...new Set(all.map((r) => r.breakdownType))].filter(Boolean).sort();
const yearList = [...new Set(all.map((r) => r.year).filter(Boolean))].sort((a, b) => a - b);

const meta = {
  generated: new Date().toISOString(),
  totalRows: all.length,
  totalDowntime: all.reduce((s, r) => s + r.downtime, 0),
  sheets: sheetStats,
  errors,
  sbus: sbuList,
  machines: machineList,
  types: typeList,
  years: yearList,
};

writeFileSync(`${DATA_DIR}\\data.json`, JSON.stringify(all));
writeFileSync(`${DATA_DIR}\\meta.json`, JSON.stringify(meta, null, 2));

console.log('\n== SUMMARY ==');
console.log('Total rows:', all.length);
console.log('Total downtime (min):', Math.round(meta.totalDowntime));
console.log('SBUs:', sbuList.join(', '));
console.log('Years:', yearList.join(', '));
console.log('Breakdown types:', typeList.length);
console.log('Errors/skipped:', errors.map((e) => e.sheet + '(' + e.error + ')').join(', ') || 'none');

// ---------- write dashboard.html (loads data at runtime) ----------
const template = readFileSync(OUT_DIR + '\\dashboard.template.html', 'utf8');
writeFileSync(OUT_DIR + '\\dashboard.html', template);
writeFileSync(OUT_DIR + '\\index.html', template);
console.log('\nWrote dashboard.html + index.html (' + (template.length / 1024).toFixed(0) + ' KB, loads data/data.json at runtime)');
