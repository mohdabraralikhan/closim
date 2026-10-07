// G9A — undo/redo history and editor session over the G8A PatternDocument.
//
// cad.ts operations are immutable: each returns a fresh document and never
// mutates its input. History therefore stores BEFORE/AFTER document
// *references* — the clone cost was already paid by the operation itself,
// so recording is free of extra copying.
//
// Gesture batching addresses the "no full-document copies per mouse
// movement" requirement: a drag opens a gesture, its per-frame operations
// only overwrite the gesture's `after` reference, and closing the gesture
// pushes exactly ONE undo entry (before = gesture start, after = last
// effective state). Intermediate documents are released for GC as the drag
// progresses because nothing retains them.
//
// Content-equal operations (e.g. moving a point to its current position)
// are not recorded, so undo can never appear to "do nothing".

import type { PatternDocument } from "../pattern/cad.js";

export interface HistoryEntry {
  label: string;
  before: PatternDocument;
  after: PatternDocument;
}

export interface HistorySnapshot {
  undoDepth: number;
  redoDepth: number;
  undoLabel: string | null;
  redoLabel: string | null;
  /** Active gesture label while a gesture is open, else null. */
  gestureLabel: string | null;
}

interface Gesture {
  label: string;
  before: PatternDocument;
  after: PatternDocument | null;
  depth: number;
}

function contentEqual(a: PatternDocument, b: PatternDocument): boolean {
  if (a === b) return true;
  return JSON.stringify(a) === JSON.stringify(b);
}

export class History {
  private undoStack: HistoryEntry[] = [];
  private redoStack: HistoryEntry[] = [];
  private gesture: Gesture | null = null;
  /** Maximum undo entries retained (oldest dropped first). */
  readonly limit: number;

  constructor(limit = 256) {
    if (!Number.isInteger(limit) || limit < 1) throw new Error(`history limit must be a positive integer, got ${limit}`);
    this.limit = limit;
  }

  /**
   * Record an applied operation. No-op when the content is unchanged.
   * While a gesture is open, entries are coalesced into the gesture.
   */
  record(label: string, before: PatternDocument, after: PatternDocument): boolean {
    if (contentEqual(before, after)) return false;
    if (this.gesture) {
      this.gesture.after = after;
      return true;
    }
    this.redoStack = [];
    this.undoStack.push({ label, before, after });
    if (this.undoStack.length > this.limit) this.undoStack.shift();
    return true;
  }

  /** Undo entries (most recent last). */
  get undoDepth(): number {
    return this.undoStack.length;
  }

  get redoDepth(): number {
    return this.redoStack.length;
  }

  get inGesture(): boolean {
    return this.gesture !== null;
  }

  /** Next undo target, or null. Reflects the open gesture's start state. */
  peekUndo(): HistoryEntry | null {
    if (this.gesture) {
      return { label: this.gesture.label, before: this.gesture.before, after: this.gesture.after ?? this.gesture.before };
    }
    return this.undoStack.length > 0 ? this.undoStack[this.undoStack.length - 1] : null;
  }

  peekRedo(): HistoryEntry | null {
    return this.redoStack.length > 0 ? this.redoStack[this.redoStack.length - 1] : null;
  }

  /**
   * Pop one undo step. Returns the entry whose `before` is the restored
   * document, or null when empty (or while a gesture is open — gestures
   * must be committed or cancelled first, never half-undone).
   */
  takeUndo(): HistoryEntry | null {
    if (this.gesture) return null;
    const entry = this.undoStack.pop();
    if (!entry) return null;
    this.redoStack.push(entry);
    return entry;
  }

  /** Pop one redo step (null when empty or gesturing). */
  takeRedo(): HistoryEntry | null {
    if (this.gesture) return null;
    const entry = this.redoStack.pop();
    if (!entry) return null;
    this.undoStack.push(entry);
    return entry;
  }

  // -------------------------------------------------------------------------
  // Gestures
  // -------------------------------------------------------------------------

  /**
   * Open a gesture starting from `current`. Subsequent record() calls
   * coalesce until endGesture(). Nesting is allowed: only the outermost
   * gesture pushes an entry.
   */
  beginGesture(label: string, current: PatternDocument): void {
    if (this.gesture) {
      this.gesture.depth++;
      return;
    }
    this.gesture = { label, before: current, after: null, depth: 0 };
  }

  /**
   * Close the gesture. Pushes one entry when effective changes occurred;
   * returns true when an entry was created.
   */
  endGesture(): boolean {
    const g = this.gesture;
    if (!g) throw new Error("endGesture: no open gesture");
    if (g.depth > 0) {
      g.depth--;
      return false;
    }
    this.gesture = null;
    if (!g.after || contentEqual(g.before, g.after)) return false;
    this.redoStack = [];
    this.undoStack.push({ label: g.label, before: g.before, after: g.after });
    if (this.undoStack.length > this.limit) this.undoStack.shift();
    return true;
  }

  /**
   * Abandon the gesture and return its starting document (the caller
   * restores session state with it). Returns null when no gesture is open.
   */
  cancelGesture(): PatternDocument | null {
    const g = this.gesture;
    if (!g) return null;
    if (g.depth > 0) {
      g.depth--;
      return null;
    }
    this.gesture = null;
    return g.before;
  }

  /** Drop all history (project open/close). */
  clear(): void {
    this.undoStack = [];
    this.redoStack = [];
    this.gesture = null;
  }

  snapshot(): HistorySnapshot {
    const u = this.peekUndo();
    const r = this.peekRedo();
    return {
      undoDepth: this.undoStack.length,
      redoDepth: this.redoStack.length,
      undoLabel: u ? u.label : null,
      redoLabel: r ? r.label : null,
      gestureLabel: this.gesture ? this.gesture.label : null,
    };
  }
}

// ---------------------------------------------------------------------------
// CadSession — document + history + selection-neutral apply helpers
// ---------------------------------------------------------------------------

type HasDocument = { document: PatternDocument };

function hasDocument(x: PatternDocument | HasDocument): x is HasDocument {
  return typeof (x as HasDocument).document === "object" && (x as HasDocument).document !== null;
}

/**
 * The editor's working state: the current PatternDocument plus its history.
 * All mutating flows go through run()/undo()/redo()/gestures so that every
 * state transition is recorded exactly once.
 *
 * This class is UI-free by design — G9D's interaction layer wraps it.
 */
export class CadSession {
  private doc: PatternDocument;
  readonly history: History;

  constructor(doc: PatternDocument, historyLimit = 256) {
    this.doc = doc;
    this.history = new History(historyLimit);
  }

  get document(): PatternDocument {
    return this.doc;
  }

  /**
   * Apply one immutable operation and record it.
   * Accepts ops returning either a document (cad.ts movePanel-style) or
   * { document, ...result } (cad.ts createPoint-style); returns whatever
   * the op returned, with the session document advanced.
   */
  run<X extends PatternDocument | HasDocument>(label: string, op: (d: PatternDocument) => X): X {
    const before = this.doc;
    const result = op(before);
    const after: PatternDocument = hasDocument(result) ? result.document : (result as PatternDocument);
    this.history.record(label, before, after);
    this.doc = after;
    return result;
  }

  /** Open a drag gesture: subsequent run() calls coalesce into one undo entry. */
  beginGesture(label: string): void {
    this.history.beginGesture(label, this.doc);
  }

  /** Close the open gesture. Returns true when an undo entry was created. */
  endGesture(): boolean {
    return this.history.endGesture();
  }

  /** True while a drag gesture is open. */
  get inGesture(): boolean {
    return this.history.inGesture;
  }

  /** Restore the gesture's start document (ESC / tool cancel). */
  cancelGesture(): boolean {    const start = this.history.cancelGesture();
    if (start === null) return false;
    this.doc = start;
    return true;
  }

  /** Undo one step. Returns the restored document, or null. */
  undo(): PatternDocument | null {
    const entry = this.history.takeUndo();
    if (!entry) return null;
    this.doc = entry.before;
    return this.doc;
  }

  /** Redo one step. Returns the restored document, or null. */
  redo(): PatternDocument | null {
    const entry = this.history.takeRedo();
    if (!entry) return null;
    this.doc = entry.after;
    return this.doc;
  }

  /**
   * Replace the document wholesale (load/import). History is cleared: the
   * previous project's states must not be reachable from a new document.
   */
  reset(doc: PatternDocument): void {
    this.doc = doc;
    this.history.clear();
  }
}
