// Deterministic ZIP archive writer (STORE method, no compression).
//
// Scope: a minimal, dependency-free ZIP 2.0 writer sufficient for bundling
// the G13 export package into a single distributable archive. Design rules:
//
//   - STORE (method 0) only: package files are small text artifacts; ZIP64,
//     compression, and encryption are out of scope and rejected up front.
//   - Fully deterministic: fixed DOS timestamp (1980-01-01 00:00:00), UTF-8
//     names, entries emitted in the order given, no extra fields. Identical
//     inputs produce byte-identical archives (asserted by tests).
//   - Integrity: CRC-32 of every entry's payload is computed and stored, so
//     any conforming unzip tool can verify the archive.
//
// This is a writer, not a general archiver: it never reads the filesystem.

const CRC_TABLE: Uint32Array = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

export function crc32(data: Uint8Array): number {
  let c = 0xffffffff;
  for (let i = 0; i < data.length; i++) c = CRC_TABLE[(c ^ data[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

export interface ZipEntry {
  /** Archive-relative path with forward slashes (no leading slash). */
  path: string;
  bytes: Uint8Array | string;
}

export interface CreateZipOptions {
  /** Comment stored in the EOCD record (ASCII; deterministic content only). */
  comment?: string;
}

const LFH_SIG = 0x04034b50;
const CDH_SIG = 0x02014b50;
const EOCD_SIG = 0x06054b50;
/** 1980-01-01 00:00:00 in DOS date/time fields — the ZIP epoch minimum. */
const DOS_TIME = 0;
const DOS_DATE = (1 << 5) | 1; // month=1, day=1 (year bits 0 => 1980)
/** General purpose bit 11: names are UTF-8. */
const UTF8_FLAG = 0x0800;
const VERSION_NEEDED = 20;
const VERSION_MADE_BY = 20;

function bytesOf(data: Uint8Array | string): Uint8Array {
  return typeof data === "string" ? new TextEncoder().encode(data) : data;
}

function asciiBytes(text: string): Uint8Array {
  const out = new Uint8Array(text.length);
  for (let i = 0; i < text.length; i++) {
    const c = text.charCodeAt(i);
    if (c > 0x7f) throw new Error(`zip: non-ASCII byte in comment at position ${i}`);
    out[i] = c;
  }
  return out;
}

class ByteWriter {
  private chunks: Uint8Array[] = [];
  private size = 0;
  private scratch = new Uint8Array(8);

  push(bytes: Uint8Array): void {
    this.chunks.push(bytes);
    this.size += bytes.length;
  }

  u8(value: number): void {
    this.scratch[0] = value & 0xff;
    this.push(this.scratch.slice(0, 1));
  }

  u16(value: number): void {
    if (!Number.isInteger(value) || value < 0 || value > 0xffff) {
      throw new Error(`zip: u16 field out of range: ${value}`);
    }
    this.scratch[0] = value & 0xff;
    this.scratch[1] = (value >>> 8) & 0xff;
    this.push(this.scratch.slice(0, 2));
  }

  u32(value: number): void {
    if (!Number.isInteger(value) || value < 0 || value > 0xffffffff) {
      throw new Error(`zip: u32 field out of range: ${value}`);
    }
    this.scratch[0] = value & 0xff;
    this.scratch[1] = (value >>> 8) & 0xff;
    this.scratch[2] = (value >>> 16) & 0xff;
    this.scratch[3] = (value >>> 24) & 0xff;
    this.push(this.scratch.slice(0, 4));
  }

  get length(): number {
    return this.size;
  }

  finish(): Uint8Array {
    const out = new Uint8Array(this.size);
    let at = 0;
    for (const chunk of this.chunks) {
      out.set(chunk, at);
      at += chunk.length;
    }
    return out;
  }
}

/**
 * Build a deterministic STORE-only ZIP archive from the given entries.
 * Entry order is preserved; duplicate paths are rejected. Throws when the
 * archive would need ZIP64 (> 65535 entries or a 4 GiB field overflow).
 */
export function createZipArchive(entries: readonly ZipEntry[], opts: CreateZipOptions = {}): Uint8Array {
  if (entries.length === 0) throw new Error("zip: refusing to write an empty archive");
  if (entries.length > 0xffff) throw new Error("zip: too many entries for a non-ZIP64 archive");
  const seen = new Set<string>();
  for (const e of entries) {
    if (e.path.length === 0 || e.path.startsWith("/") || e.path.includes("\\")) {
      throw new Error(`zip: invalid entry path '${e.path}'`);
    }
    if (seen.has(e.path)) throw new Error(`zip: duplicate entry path '${e.path}'`);
    seen.add(e.path);
    if (!Number.isFinite(e.bytes.length) || e.bytes.length < 0) {
      throw new Error(`zip: entry '${e.path}' has an invalid size`);
    }
  }

  const out = new ByteWriter();
  interface Central {
    name: Uint8Array;
    crc: number;
    size: number;
    offset: number;
  }
  const central: Central[] = [];

  for (const entry of entries) {
    const name = new TextEncoder().encode(entry.path);
    if (name.length > 0xffff) throw new Error(`zip: entry name too long '${entry.path}'`);
    const data = bytesOf(entry.bytes);
    if (data.length > 0xffffffff) throw new Error(`zip: entry '${entry.path}' exceeds 4 GiB (ZIP64 required)`);
    const crc = crc32(data);
    const offset = out.length;
    if (offset > 0xffffffff) throw new Error("zip: archive exceeds 4 GiB (ZIP64 required)");

    // Local file header.
    out.u32(LFH_SIG);
    out.u16(VERSION_NEEDED);
    out.u16(UTF8_FLAG);
    out.u16(0); // method STORE
    out.u16(DOS_TIME);
    out.u16(DOS_DATE);
    out.u32(crc);
    out.u32(data.length); // compressed size
    out.u32(data.length); // uncompressed size
    out.u16(name.length);
    out.u16(0); // extra field length
    out.push(name);
    out.push(data);

    central.push({ name, crc, size: data.length, offset });
  }

  // Central directory.
  const cdStart = out.length;
  for (const c of central) {
    out.u32(CDH_SIG);
    out.u16(VERSION_MADE_BY);
    out.u16(VERSION_NEEDED);
    out.u16(UTF8_FLAG);
    out.u16(0); // method STORE
    out.u16(DOS_TIME);
    out.u16(DOS_DATE);
    out.u32(c.crc);
    out.u32(c.size);
    out.u32(c.size);
    out.u16(c.name.length);
    out.u16(0); // extra
    out.u16(0); // comment length
    out.u16(0); // disk number start
    out.u16(0); // internal attrs
    out.u32(0); // external attrs
    out.u32(c.offset);
    out.push(c.name);
  }
  const cdSize = out.length - cdStart;

  // End of central directory.
  const commentBytes = opts.comment !== undefined ? asciiBytes(opts.comment) : new Uint8Array(0);
  if (commentBytes.length > 0xffff) throw new Error("zip: comment too long");
  out.u32(EOCD_SIG);
  out.u16(0); // this disk
  out.u16(0); // disk with CD
  out.u16(central.length);
  out.u16(central.length);
  out.u32(cdSize);
  out.u32(cdStart);
  out.u16(commentBytes.length);
  out.push(commentBytes);

  return out.finish();
}
