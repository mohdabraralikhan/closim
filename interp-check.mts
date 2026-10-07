import { readFileSync } from 'node:fs';
import { importApparelDxf } from './src/cad/dxf-import.js';
const text = readFileSync(process.argv[2], 'utf8');
const r = importApparelDxf(text, { profile: 'aama-style' });
console.log(JSON.stringify({
  ok: r.ok, units: r.units, pieces: r.pieces.length,
  mismatches: r.mismatches, layersFound: r.layersFound,
}));
