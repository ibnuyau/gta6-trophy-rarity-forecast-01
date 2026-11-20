// GTA VI trophy rarity forecast (make.fun gta6-trophy-rarity-forecast-01).
//
// Model: saturating logit curves per trophy, fitted on Exophase's refresh points.
//
// Why this shape. A trophy's earn rate is earners / players-tracked, and both move:
// Exophase refreshes its counts on its own schedule (hours apart), so the page sits
// unchanged between refreshes and jumps at them; new players joining dilute every
// rate, while the players already tracked keep earning trophies. On launch night
// the first refresh took players-tracked 638 -> 3,868 while total trophies earned
// went 943 -> 4,276: almost all of the new earns landed on the one trophy already
// at 100%, so every other rate fell to roughly a sixth of its launch-night value.
// A forecaster that repeats the last rate (the house baseline) or extends a
// straight line in logit space (the house curve baseline) misses both effects:
// the dilution at each refresh and the saturation as players finish the early
// game and the casual tail arrives.
//
// What the program does, per trophy:
//   1. Collapse the history to refresh points: a read only counts when the
//      players-tracked count or any rate changed, so the long flat stretches
//      between Exophase refreshes cannot outweigh the jumps.
//   2. Fit logit(rate) against ln(1 + hours since release) by weighted least
//      squares, with a ridge-penalised quadratic term: the linear part is the
//      house curve baseline's shape (a rate that rises fast, then ever more
//      slowly, in log time); the quadratic part lets a trophy bend over and
//      saturate, or keep falling, when the data says so. Recent refreshes weigh
//      more (36 h half-life in weight terms). Each trophy is fitted on its own:
//      shrinking slopes toward the cross-trophy median was tried and hurt, so
//      there is no shrinkage.
//   3. Read the fit at the target, pulled back by half the median refresh gap:
//      the answer is the page as of Exophase's last refresh before the target,
//      not the target instant itself.
//   4. If the history is flat (no refresh seen yet, as on launch night), fall
//      back to a dilution prior: expect the next refresh to cut rates to about
//      a quarter, with a wide spread -- the first refresh cut most rates to
//      about a sixth, and a later refresh dilutes less as the player influx
//      slows.
//   5. Quantiles from a logit-normal spread that combines a random-walk term
//      growing with the horizon, the fit's residual, a fixed allowance for one
//      unmodelled refresh jump, and count noise for rare trophies. The spread
//      is honest: it widens with horizon and with how little the trophy's own
//      history says.
//
// The program is deterministic (no randomness, no clock reads, no network) and
// reads $MAKEFUN_INPUT_DIR / $MAKEFUN_OUTPUT_DIR, else /input / /output.
// Node 22, standard library only.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** Standard normal quantiles at those levels. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];

/** Release day, the origin of the log-time axis (same as the house baselines). */
const RELEASE_MS = Date.parse('2026-11-19T00:00:00Z');

/** Recency half-life for weighting refresh points, in hours. */
const RECENCY_HALF_LIFE_H = 36;
/** Ridge penalty on the quadratic coefficient (logit per log-time^2). */
const RIDGE_QUAD = 0.05;
/** Largest logit move from the last read the forecast may claim. */
const MAX_LOGIT_MOVE = 2.5;
/** Random-walk spread, logit units per square-root day. */
const RW_SIGMA = 0.15;
/** Fixed allowance for one unmodelled refresh jump, logit units. */
const JUMP_SIGMA = 0.2;
/** Default refresh gap (hours) when no refresh is visible in the history. */
const DEFAULT_GAP_H = 5;
/** Flat-history dilution: one refresh multiplies a rate by about this. */
const FLAT_DILUTION = 0.18;
/** Flat-history spread, logit units. */
const FLAT_SIGMA = 0.8;

interface Trophy {
  id: number;
}
interface Read {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}
interface Case {
  levels: number[];
  horizon_minutes: number;
  target: string;
  at: string;
  trophies: Trophy[];
  history: Read[];
}

const clampRate = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const p = clampRate(pct) / 100;
  return Math.log(p / (1 - p));
};
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;
const hoursSinceRelease = (iso: string): number =>
  Math.max(0, (Date.parse(iso) - RELEASE_MS) / 3_600_000);
const xOf = (iso: string): number => Math.log(1 + hoursSinceRelease(iso));

/** Solve a small symmetric positive-definite system by Gaussian elimination. */
function solve3(a: number[][], b: number[]): number[] {
  const n = 3;
  const m = a.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(m[r][col]) > Math.abs(m[piv][col])) piv = r;
    [m[col], m[piv]] = [m[piv], m[col]];
    const d = m[col][col] || 1e-12;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = m[r][col] / d;
      for (let k = col; k <= n; k++) m[r][k] -= f * m[col][k];
    }
  }
  return m.map((row, i) => row[n] / (row[i] || 1e-12));
}

/** Weighted least squares for y ~ a + b*x. */
function fitLine(
  pts: Array<{ x: number; y: number; w: number }>,
): { a: number; b: number; resid: number } {
  let sw = 0;
  let sx = 0;
  let sy = 0;
  let sxx = 0;
  let sxy = 0;
  for (const p of pts) {
    sw += p.w;
    sx += p.w * p.x;
    sy += p.w * p.y;
    sxx += p.w * p.x * p.x;
    sxy += p.w * p.x * p.y;
  }
  const denom = sw * sxx - sx * sx;
  const b = denom !== 0 ? (sw * sxy - sx * sy) / denom : 0;
  const a = (sy - b * sx) / Math.max(sw, 1e-9);
  let ss = 0;
  for (const p of pts) ss += p.w * (p.y - (a + b * p.x)) ** 2;
  return { a, b, resid: Math.sqrt(ss / Math.max(sw, 1e-9)) };
}

/** Weighted least squares for y ~ a + b*x + c*x^2, ridge on c only. */
function fitQuad(
  pts: Array<{ x: number; y: number; w: number }>,
): { a: number; b: number; c: number; resid: number } {
  const s = [0, 0, 0, 0, 0, 0, 0, 0, 0];
  const t = [0, 0, 0];
  for (const p of pts) {
    const xp = [1, p.x, p.x * p.x];
    for (let i = 0; i < 3; i++) {
      t[i] += p.w * xp[i] * p.y;
      for (let j = 0; j < 3; j++) s[i * 3 + j] += p.w * xp[i] * xp[j];
    }
  }
  s[8] += RIDGE_QUAD;
  const [a, b, c] = solve3(
    [
      [s[0], s[1], s[2]],
      [s[3], s[4], s[5]],
      [s[6], s[7], s[8]],
    ],
    t,
  );
  let ss = 0;
  let sw = 0;
  for (const p of pts) {
    ss += p.w * (p.y - (a + b * p.x + c * p.x * p.x)) ** 2;
    sw += p.w;
  }
  return { a, b, c, resid: Math.sqrt(ss / Math.max(sw, 1e-9)) };
}

function medianOf(xs: number[]): number {
  const s = [...xs].sort((p, q) => p - q);
  const m = s.length;
  return m === 0 ? 0 : m % 2 === 1 ? s[(m - 1) / 2] : (s[m / 2 - 1] + s[m / 2]) / 2;
}

function forecastCase(c: Case): { trophies: Array<{ id: number; q: number[] }> } {
  const horizonDays = c.horizon_minutes / 1440;
  const last = c.history[c.history.length - 1];

  // Collapse the history to refresh points: the reads at which anything moved.
  // Keep, per trophy, the (x, logit rate) pairs at those points.
  const nT = c.trophies.length;
  const points: Array<Array<{ x: number; y: number; t: number }>> = Array.from(
    { length: nT },
    () => [],
  );
  const gapHours: number[] = [];
  let nRefreshes = 0;
  let prevRates: Array<number | null> | null = null;
  let prevPlayers: number | null = null;
  let prevAt: string | null = null;
  for (const h of c.history) {
    let changed = prevRates === null;
    if (!changed) {
      if (h.players_tracked !== prevPlayers) changed = true;
      else {
        // Only a move between two real rates counts: a trophy appearing or
        // disappearing (null <-> value) is not an Exophase refresh.
        for (let i = 0; i < nT; i++) {
          if (h.rates[i] != null && prevRates![i] != null && h.rates[i] !== prevRates![i]) {
            changed = true;
            break;
          }
        }
      }
    }
    if (changed) {
      nRefreshes++;
      if (prevAt !== null) gapHours.push((Date.parse(h.at) - Date.parse(prevAt)) / 3_600_000);
      const x = xOf(h.at);
      const t = hoursSinceRelease(h.at);
      for (let i = 0; i < nT; i++) {
        if (h.rates[i] != null) points[i].push({ x, y: logit(h.rates[i]!), t });
      }
      prevAt = h.at;
    }
    prevRates = h.rates;
    prevPlayers = h.players_tracked;
  }
  const gapH = gapHours.length > 0 ? medianOf(gapHours) : DEFAULT_GAP_H;
  // The answer is the page as of the last refresh before the target.
  const xEff = Math.log(1 + Math.max(0, hoursSinceRelease(c.target) - gapH / 2));

  const rows: Array<{ id: number; q: number[] }> = [];
  for (let i = 0; i < nT; i++) {
    const lastRate = last.rates[i] ?? 0;
    const lastL = logit(lastRate);
    let mu: number;
    let sigma: number;
    const pts = points[i];

    if (lastRate >= 99.99) {
      // A trophy every tracked player holds stays (near) 100%: the first
      // refresh left the launch-night 100% trophy at 100%.
      mu = logit(99.99);
      sigma = 0.05;
    } else if (pts.length < 2 && nRefreshes <= 1) {
      // Flat history: no refresh seen yet anywhere in the case (launch
      // night). Expect the next refresh to dilute the rate (players join
      // faster than earners), with a wide spread.
      const refreshes = Math.max(1, c.horizon_minutes / 60 / DEFAULT_GAP_H);
      const dil = Math.pow(FLAT_DILUTION, Math.min(refreshes, 3));
      mu = logit(clampRate(lastRate * dil));
      sigma = FLAT_SIGMA;
    } else if (pts.length < 2) {
      // Refreshes happened, but this trophy's rate never moved: its
      // dilution and progression have balanced so far. Persist it, with a
      // spread for the next refreshes.
      mu = lastL;
      const predRate = invLogit(mu);
      const countNoise = 0.5 / Math.sqrt(Math.max(0.2, predRate * 20));
      sigma = Math.min(
        2.0,
        Math.sqrt(
          (RW_SIGMA * Math.sqrt(horizonDays)) ** 2 + 0.2 ** 2 + JUMP_SIGMA ** 2 + countNoise ** 2,
        ),
      );
    } else {
      const tLast = pts[pts.length - 1].t;
      const wpts = pts.map((p) => ({
        x: p.x,
        y: p.y,
        w: Math.pow(0.5, (tLast - p.t) / RECENCY_HALF_LIFE_H),
      }));
      const q = fitQuad(wpts);
      const lin = fitLine(wpts);
      const predQuad = q.a + q.b * xEff + q.c * xEff * xEff;
      const predLin = lin.a + lin.b * xEff;
      // Trust the bend only when there is enough data and it clearly helps.
      let sseLin = 0;
      let sseQuad = 0;
      for (const p of wpts) {
        sseLin += p.w * (p.y - (lin.a + lin.b * p.x)) ** 2;
        sseQuad += p.w * (p.y - (q.a + q.b * p.x + q.c * p.x * p.x)) ** 2;
      }
      const useQuad = pts.length >= 6 && sseQuad < 0.85 * sseLin;
      mu = useQuad ? 0.5 * predQuad + 0.5 * predLin : predLin;
      // Never claim a move larger than the clamp from the last read.
      mu = Math.min(lastL + MAX_LOGIT_MOVE, Math.max(lastL - MAX_LOGIT_MOVE, mu));
      const resid = Math.min(0.8, Math.max(0.15, q.resid));
      const predRate = invLogit(mu);
      const countNoise = 0.5 / Math.sqrt(Math.max(0.2, predRate * 20));
      sigma = Math.min(
        2.0,
        Math.sqrt(
          (RW_SIGMA * Math.sqrt(horizonDays)) ** 2 + resid ** 2 + JUMP_SIGMA ** 2 + countNoise ** 2,
        ),
      );
    }
    sigma = Math.max(0.05, Math.min(2.0, sigma));
    const q = Z.map((z) => round4(clampRate(invLogit(mu + z * sigma))));
    // Keep the row non-decreasing after rounding/clamping.
    for (let j = 1; j < q.length; j++) if (q[j] < q[j - 1]) q[j] = q[j - 1];
    rows.push({ id: c.trophies[i].id, q });
  }
  return { trophies: rows };
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR)
  .filter((f) => f.endsWith('.in'))
  .sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (
    c.levels.length !== LEVELS.length ||
    c.levels.some((t, j) => t !== LEVELS[j])
  ) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  if (c.history.length === 0) throw new Error(`${file} has no history`);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify(forecastCase(c))}\n`);
}
