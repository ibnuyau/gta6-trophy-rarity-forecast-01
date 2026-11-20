# GTA VI trophy rarity forecast — saturating logit curves, honest quantiles

[![gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ibnuyau/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ibnuyau/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Entry for the [make.fun bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01):
forecast, for every Grand Theft Auto VI PlayStation 5 trophy, its earn rate on
Exophase at 2026-11-26T00:00:00Z, as five quantiles (5%, 25%, 50%, 75%, 95%).
`main.ts` (Node 22, standard library only) reads each case in `/input` and
writes `/output/<id>.out`.

## Model

A trophy's earn rate is earners / players-tracked, and both move. Exophase
refreshes its counts on its own schedule (hours apart): the page sits unchanged
between refreshes and jumps at them. New players joining dilute every rate,
while the players already tracked keep earning trophies. On launch night the
first refresh took players-tracked 638 -> 3,868 while total trophies earned went
943 -> 4,276, and almost all of the new earns landed on the one trophy already
at 100%, so every other rate fell to roughly a sixth. A forecaster that repeats
the last rate (the house persistence baseline) or extends a straight line in
logit space (the house curve baseline) misses both effects: the dilution at
each refresh and the saturation as players finish the early game and the
casual tail arrives.

Per trophy, the program:

- Collapses the history to refresh points: a read only counts when the
  players-tracked count or a rate changed, so the long flat stretches between
  Exophase refreshes cannot outweigh the jumps.
- Fits logit(rate) against ln(1 + hours since release) by weighted least
  squares, with a ridge-penalised quadratic term. The linear part is the house
  curve baseline's shape; the quadratic part lets a trophy bend over and
  saturate, or keep falling, when the data says so (used only with 6+ refresh
  points and a clear fit improvement, blended 50/50 with the linear fit).
  Recent refreshes weigh more (36 h half-life). Each trophy is fitted on its
  own: shrinking slopes toward the cross-trophy median was tried and hurt.
- Reads the fit at the target, pulled back by half the median refresh gap:
  the answer is the page as of Exophase's last refresh before the target.
- If the history is flat (no refresh seen yet, as on launch night), falls back
  to a dilution prior: the next refresh cuts rates to about 0.18x, with a wide
  spread. A trophy whose rate never moved while other trophies did is
  persisted instead: its dilution and progression have balanced so far.
- Quantiles come from a logit-normal spread combining a random-walk term
  growing with the horizon, the fit's residual, an allowance for one
  unmodelled refresh jump, and count noise for rare trophies.

Deterministic (no randomness, no clock reads, no network), about 0.09 s per
case, well inside the 8 s cap.

## Backtests

The practice cases' true answers are sealed server-side, so the practice
backtest reconstructs them from the facts the rules publish (players 638 ->
3,868, trophies earned 943 -> 4,276, every rate but the 100% one fell, and the
house baselines score about 1.51 loss / 0.68 points typical miss on them):

- Scenario B, near-zero new earners per trophy: this entry 0.175, house
  persistence 1.688, house curve 1.688.
- Scenario C, the ~103 leftover new earns land on the five easiest trophies
  (this reconstruction reproduces the house baselines at 1.581 loss and 0.677
  typical miss, close to the published ~1.51 / ~0.68): this entry 0.213,
  house persistence 1.581, house curve 1.581.
- Scenario A, proportional dilution (new players earn in the old mix): this
  entry 0.828, house baselines 0.203. This scenario contradicts the published
  house practice figures, so it is the unlikely one; it is reported for
  honesty, not hidden.

Final-like simulation: 18 synthetic final cases (two days of 10-minute reads,
refreshes every 3-6 h, 4.5-day horizon) across six trajectory families
(steady rise, dilution wins, mixed, weekend surge after the checkpoint, hard
saturation, slope flip mid-course), each with three seeds. Mean pinball loss:
this entry 0.144, house persistence 0.366, house curve 0.234. The entry beats
both house baselines in every family.

## Run locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

## License

MIT (see LICENSE).
