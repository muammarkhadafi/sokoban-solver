// Sokoban solver: room decomposition + push-optimal A* per room.
//
// A level is a hallway with 1-4 rooms off it (3-4 boxes each). A box pushed out of a
// room into the hallway can never come back, so every room is an independent problem:
// solve each room on its own, walk the hallway between rooms, and the push counts add
// up to the level's push optimum. Each room is solved with A* over pushes (a state is
// the boxes' squares plus the player's reachable region, named by its lowest square),
// with an admissible Hungarian heuristic (minimum box-goal matching over walk
// distances), dead-square pruning and frozen 2x2 pruning. The heuristic is admissible,
// so the first solved state popped is push-optimal for that room.
//
// Determinism: every budget is a count of work (state expansions), never the clock.
// No Date, no Math.random, no environment reads. Two runs write identical bytes.
//
// Local run: MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The four moves: [dx, dy, walk letter, push letter]. */
const DIRS: readonly (readonly [number, number, string, string])[] = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

/** Most expansions one room search may spend (A*, then the greedy retry). */
const MAX_EXPANSIONS_PER_ROOM = 250_000;
/** Most expansions the whole-level fallback may spend. */
const MAX_EXPANSIONS_WHOLE = 150_000;
/** Most expansions the whole run (all 5 instances) may spend, so a pathological set can never eat the wall budget. */
const RUN_EXPANSION_BUDGET = 3_000_000;

const budget = { used: 0 };

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

interface Room {
  boxes: number[];
  goals: number[];
}

/** Parse XSB text. Short rows and the outer frame count as walls. */
function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  return { w, h, wall, goals: goals.sort((a, b) => a - b), boxes: boxes.sort((a, b) => a - b), player };
}

/** Squares from which a box alone can be pushed onto some goal (pulled back from the goals over open floor). */
function liveSquares(level: Level): Uint8Array {
  const { w, wall, goals } = level;
  const live = new Uint8Array(wall.length);
  const queue: number[] = [];
  for (const g of goals) {
    live[g] = 1;
    queue.push(g);
  }
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi]!;
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (!wall[y] && !wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

/** Walk distances from `from` over open floor (-1 where unreachable). */
function bfsDist(wall: Uint8Array, w: number, from: number): Int32Array {
  const dist = new Int32Array(wall.length).fill(-1);
  const queue: number[] = [from];
  dist[from] = 0;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    for (const [dx, dy] of DIRS) {
      const q = p + dy * w + dx;
      if (wall[q] === 0 && dist[q] < 0) {
        dist[q] = dist[p]! + 1;
        queue.push(q);
      }
    }
  }
  return dist;
}

/** The shortest walk from `from` to `to` around walls and boxes (lower LURD letters, first direction first on ties), or null. */
function walkTo(level: Level, occupied: Uint8Array, from: number, to: number): string | null {
  const { w, wall } = level;
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array<string>(wall.length);
  const queue: number[] = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) return null;
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

/** The player's square after playing `moves` (case aside) from `from` on a board `w` wide. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

/**
 * Push-optimal search over one (sub-)level: A* over pushes, or greedy best-first when
 * `mode` is 'greedy'. A state is the sorted boxes plus the player's region (named by its
 * lowest square). Prunes dead squares and frozen 2x2 blocks. Returns {pushes, moves} or
 * null when the budget runs out or no solution exists. The moves are lower letters for
 * walks and upper letters for pushes.
 */
function solvePush(level: Level, maxExpansions: number, mode: 'astar' | 'greedy'): { pushes: number; moves: string } | null {
  const { w, wall, goals } = level;
  const size = wall.length;
  const n = level.boxes.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveSquares(level);
  for (const b of level.boxes) if (!live[b]) return null;

  // Walk distance from each goal (Hungarian needs every box-goal pair).
  const distToGoal = goals.map((g) => bfsDist(wall, w, g));
  // All bijections of boxes onto goals, as goal-index orders (n <= 6 here; larger falls back to sum-of-min).
  const perms: number[][] =
    n <= 6
      ? (() => {
          const out: number[][] = [];
          const cur: number[] = [];
          const used = new Array<boolean>(n).fill(false);
          const rec = (): void => {
            if (cur.length === n) {
              out.push(cur.slice());
              return;
            }
            for (let i = 0; i < n; i++) {
              if (used[i]) continue;
              used[i] = true;
              cur.push(i);
              rec();
              cur.pop();
              used[i] = false;
            }
          };
          rec();
          return out;
        })()
      : [];
  /** Admissible estimate of pushes left: Hungarian matching, or sum of nearest-goal distances. */
  const hOf = (boxes: number[]): number => {
    if (perms.length > 0) {
      let best = Infinity;
      for (const perm of perms) {
        let sum = 0;
        for (let i = 0; i < n; i++) {
          const d = distToGoal[perm[i]!]![boxes[i]!]!;
          if (d < 0) {
            sum = Infinity;
            break;
          }
          sum += d;
        }
        if (sum < best) best = sum;
      }
      return best;
    }
    let sum = 0;
    for (const b of boxes) {
      let m = Infinity;
      for (let g = 0; g < goals.length; g++) {
        const d = distToGoal[g]![b]!;
        if (d >= 0 && d < m) m = d;
      }
      if (m === Infinity) return Infinity;
      sum += m;
    }
    return sum;
  };

  // State keys: base-`size` digits (low, then the boxes) while they stay exact, else strings.
  const numericKey = Math.pow(size, n + 1) < Number.MAX_SAFE_INTEGER;
  const keyOf = (boxes: number[], low: number): number | string => {
    if (numericKey) {
      let k = low;
      for (let i = 0; i < boxes.length; i++) k = k * size + boxes[i]!;
      return k;
    }
    return low + '|' + boxes.join(',');
  };

  // Flood fill: the squares reachable from `from` with `occupied` in the way, and the lowest of them.
  const seenA = new Int32Array(size);
  const seenB = new Int32Array(size);
  const stack = new Int32Array(size);
  let stampA = 0;
  let stampB = 0;
  const flood = (seen: Int32Array, stamp: number, occupied: Uint8Array, from: number): number => {
    let low = from;
    let sp = 0;
    stack[sp++] = from;
    seen[from] = stamp;
    while (sp > 0) {
      const p = stack[--sp]!;
      if (p < low) low = p;
      const q0 = p - w;
      if (wall[q0] === 0 && occupied[q0] === 0 && seen[q0] !== stamp) {
        seen[q0] = stamp;
        stack[sp++] = q0;
      }
      const q1 = p + w;
      if (wall[q1] === 0 && occupied[q1] === 0 && seen[q1] !== stamp) {
        seen[q1] = stamp;
        stack[sp++] = q1;
      }
      const q2 = p - 1;
      if (wall[q2] === 0 && occupied[q2] === 0 && seen[q2] !== stamp) {
        seen[q2] = stamp;
        stack[sp++] = q2;
      }
      const q3 = p + 1;
      if (wall[q3] === 0 && occupied[q3] === 0 && seen[q3] !== stamp) {
        seen[q3] = stamp;
        stack[sp++] = q3;
      }
    }
    return low;
  };
  /** Whether the box just pushed to `t` sits in a frozen 2x2 block of walls and boxes with a box off its goal. */
  const frozenBlock = (occupied: Uint8Array, t: number): boolean => {
    for (const a of [t, t - 1, t - w, t - w - 1]) {
      const b = a + 1;
      const c = a + w;
      const d = c + 1;
      if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b]) || !(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
      if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b]) || (occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) return true;
    }
    return false;
  };

  const solved = (boxes: number[]): boolean => boxes.every((b) => goal[b]);
  const occupied = new Uint8Array(size);
  for (const b of level.boxes) occupied[b] = 1;
  const startH = hOf(level.boxes);
  if (startH === Infinity) return null;
  if (solved(level.boxes)) return { pushes: 0, moves: '' };

  // States: boxes, the player's square after the push that made them, parent, the push, and its key.
  const sBoxes: number[][] = [level.boxes.slice()];
  const sPlayer: number[] = [level.player];
  const sParent: number[] = [-1];
  const sFrom: number[] = [-1];
  const sTo: number[] = [-1];
  const sG: number[] = [0];
  const sKey: (number | string)[] = [keyOf(level.boxes, flood(seenA, ++stampA, occupied, level.player))];
  for (const b of level.boxes) occupied[b] = 0;
  const best = new Map<number | string, number>([[sKey[0]!, 0]]);

  // Binary heap on [priority, g, insertion order, state]: A* orders by f = g + h (ties: larger g first),
  // greedy orders by h alone. The order is fixed, so two runs agree.
  const heap: [number, number, number, number][] = [];
  const less = (a: [number, number, number, number], b: [number, number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] > b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: [number, number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): [number, number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };

  let seq = 0;
  heapPush([mode === 'astar' ? startH : startH, 0, seq++, 0]);
  let expanded = 0;
  while (heap.length > 0) {
    const [, g, , s] = heapPop();
    if (best.get(sKey[s]!) !== s) continue; // a better path to this state has been found since
    if (solved(sBoxes[s]!)) {
      // Rebuild the moves: walk to each push square, then push.
      const path: number[] = [];
      for (let t = s; t > 0; t = sParent[t]!) path.push(t);
      path.reverse();
      let moves = '';
      let player = level.player;
      let boxes = level.boxes.slice();
      for (const t of path) {
        const from = sFrom[t]!;
        const to = sTo[t]!;
        const d = to - from;
        const dir = DIRS.find(([dx, dy]) => dy * w + dx === d)!;
        for (const b of boxes) occupied[b] = 1;
        const walk = walkTo(level, occupied, player, from - d);
        for (const b of boxes) occupied[b] = 0;
        if (walk === null) return null;
        moves += walk + dir[3];
        player = from;
        const i = boxes.indexOf(from);
        boxes[i] = to;
        boxes.sort((a, b) => a - b);
      }
      return { pushes: path.length, moves };
    }
    if (expanded >= maxExpansions || budget.used >= RUN_EXPANSION_BUDGET) return null;
    expanded++;
    budget.used++;
    const boxes = sBoxes[s]!;
    const player = sPlayer[s]!;
    for (const b of boxes) occupied[b] = 1;
    const regionStamp = ++stampA;
    flood(seenA, regionStamp, occupied, player);
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi]!;
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx;
        const behind = b - d;
        const t = b + d;
        if (seenA[behind] !== regionStamp) continue;
        if (wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(occupied, t);
        let key: number | string = -1;
        let nb: number[] | null = null;
        let nh = Infinity;
        if (!frozen) {
          nb = boxes.slice();
          nb[bi] = t;
          nb.sort((a, b2) => a - b2);
          nh = hOf(nb);
          if (nh !== Infinity) key = keyOf(nb, flood(seenB, ++stampB, occupied, b));
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen || nh === Infinity) continue;
        const prev = best.get(key);
        if (prev !== undefined && sG[prev]! <= g + 1) continue;
        const idx = sBoxes.length;
        sBoxes.push(nb!);
        sPlayer.push(b);
        sParent.push(s);
        sFrom.push(b);
        sTo.push(t);
        sG.push(g + 1);
        sKey.push(key);
        best.set(key, idx);
        const h = nh;
        heapPush([mode === 'astar' ? g + 1 + h : h, g + 1, seq++, idx]);
      }
    }
    for (const b of boxes) occupied[b] = 0;
  }
  return null;
}

/**
 * The level's rooms: cut the dead squares of the player's row (the hallway: a box there
 * reaches no goal) out of the floor; each part left that holds a box is a room. Null when
 * some part holds more or fewer boxes than goals, a goal falls outside every part, or a
 * box starts in the hallway: then the level is no set of rooms.
 */
function roomsOf(level: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(level);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let x = 0; x < w; x++) {
    const i = row * w + x;
    if (!wall[i] && !live[i]) cut[i] = 1;
  }
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i]! + dy * w + dx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ boxes: [], goals: [] });
  }
  for (const b of boxes) rooms[part[b]!]!.boxes.push(b);
  for (const g of goals) {
    if (part[g]! < 0) return null;
    rooms[part[g]!]!.goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/** The level solved room by room: each room's push-optimal solution in turn, the other rooms' boxes standing as walls. */
function solveByRooms(level: Level): string | null {
  const rooms = roomsOf(level);
  if (rooms === null) return null;
  let player = level.player;
  let moves = '';
  const done = new Set<number>();
  for (let ri = 0; ri < rooms.length; ri++) {
    const room = rooms[ri]!;
    const wall = level.wall.slice();
    for (let j = 0; j < rooms.length; j++) {
      if (j === ri) continue;
      const other = rooms[j]!;
      for (const b of done.has(j) ? other.goals : other.boxes) wall[b] = 1;
    }
    const sub: Level = {
      w: level.w,
      h: level.h,
      wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    let solved = solvePush(sub, MAX_EXPANSIONS_PER_ROOM, 'astar');
    if (solved === null) solved = solvePush(sub, MAX_EXPANSIONS_PER_ROOM, 'greedy');
    if (solved === null) return null;
    moves += solved.moves;
    player = endOf(level.w, player, solved.moves);
    done.add(ri);
  }
  return moves;
}

/** Solve one level: rooms first, the whole level as a fallback, an empty answer if neither works. */
function solveLevel(level: Level): string {
  const byRooms = solveByRooms(level);
  if (byRooms !== null) return byRooms;
  const whole = solvePush(level, MAX_EXPANSIONS_WHOLE, 'astar') ?? solvePush(level, MAX_EXPANSIONS_WHOLE, 'greedy');
  return whole === null ? '' : whole.moves;
}

for (const id of readdirSync(I).sort()) {
  const moves = solveLevel(parseLevel(readFileSync(join(I, id, 'level.txt'), 'utf8')));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
