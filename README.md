# Reliable temporal rule discovery for financial event streams

This repository discovers small sets of temporal rules from multivariate
financial event histories. A rule records which events occurred, the time in
which they formed, and whether their fitted effect raises or lowers the target
intensity. Discovery uses the fit partition. Independent reliability checks
use the certification partition. The test partition is evaluated once after
all rule definitions and model choices have been fixed.

The frozen experiments cover two settings.

- Aave V2 and V3 borrower liquidations on Ethereum
- WSELOB-2017 volatility bursts for PEKAO, KGHM, and PKNORLEN

Raw data, processed data, paper sources, notes, logs, checkpoints, and results
are intentionally excluded from Git. No API key, wallet identifier, local
path, author name, or affiliation is required by the tracked source tree.

## Repository layout

```text
configs/experiments/   final rule-discovery configurations
configs/baselines/     comparison-model configurations
native/                CPU and CUDA kernels
src/crbstpp/           model, search, certification, and preprocessing code
tests/crbstpp/         unit and regression tests
tools/                 evaluation and reproduction helpers
```

## Environment

Python 3.10 or newer is required. The published configurations use two NVIDIA
GPUs. The CUDA extension targets compute capability 8.9 by default.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[baselines,test]'
```

The same environment can be created with Docker.

```bash
cp .env.example .env
docker compose build
docker compose run --rm app bash -lc \
  "python -m pip install -e '.[baselines,test]' && exec bash"
```

Store private RPC endpoints only in `.env`. They must never be added to a YAML
configuration or committed.

## Download the raw data

The frozen raw observation ends at Ethereum block `25,660,939`. Downloading
requires an archive-capable Ethereum RPC endpoint. If `AAVE_RPC_URL` is not
set, the downloader tries its public defaults.

```bash
tools/reproduce_experiments.sh download
```

This command also downloads PEKAO, KGHM, and PKN Orlen from WSELOB-2017.
The WSELOB files are distributed through Mendeley Data under CC BY 4.0.
Downloaded files remain under `data/` and are never tracked by Git.

## Reproduce the experiments

The paper reports quantitative results over three independent runs. The
reproduction script uses the same three runs by default. It rebuilds each data
partition, discovers and certifies rules, fits every baseline, evaluates the
fixed test partition once, and writes one metric report per run.

```bash
tools/reproduce_experiments.sh all
```

The random seeds are 111, 222, and 333. They are fixed before model fitting.
The paper aggregates prediction metrics over all three runs. Its qualitative
rule tables use run 111 so that every stated rule maps to one fixed discovery
and certification split. To reproduce only one run while debugging, use:

```bash
tools/reproduce_experiments.sh all --seed 111
```

The script can also execute individual stages without changing their order or
settings.

```bash
tools/reproduce_experiments.sh prepare
tools/reproduce_experiments.sh rules
tools/reproduce_experiments.sh baselines
tools/reproduce_experiments.sh metrics
```

## Rules reported from run 111

The direction of every certified rule below is excitation. It means that the
pattern predicts a higher subsequent target intensity. It is not a causal
claim about the individual action.

### Aave liquidation

- A1 combines variable borrowing from a previously used debt reserve with a
  third-party collateral top-up within 23 epochs. It describes continued
  borrowing while another wallet supplies collateral.
- A2 adds entry into variable debt backed by a stable-value asset within 26
  epochs. It combines renewed leverage, external support, and a change in debt
  composition.
- A3 combines collateral supplied by the borrower and by another wallet within
  23 epochs. It identifies simultaneous intervention by multiple parties.
- A4 is a two-rule set, not an event conjunction. One rule is a third-party
  collateral top-up. The other is the last collateral withdrawal while debt
  remains. Their joint inclusion shows that external support and collateral
  removal provide separate warning information.
- A5 combines a third-party collateral top-up with collateral reserve
  activation within 24 epochs. It distinguishes collateral reconfiguration
  from an undifferentiated deposit.

Third-party collateral support appears in every certified Aave rule set. The
result does not say that support causes liquidation. It is a reliable marker
of positions that already require intervention. The additional borrowing and
collateral events distinguish the setting in which that marker appears.

Six plausible Aave candidates were not certified. Collateral rotation with
third-party repayment and mixed stable- and variable-rate borrowing did not
reproduce a clear held-out improvement. New-reserve borrowing, entry into
volatile-asset debt, self-repayment with external support, and collateral
reserve deactivation were not robust across held-out positions. These failures
mean that the evidence was insufficient for a reliable warning rule, not that
the financial patterns are impossible.

### WSELOB volatility

- W1 contains two rules. The first combines balance-restoring passive liquidity
  addition, an imbalance-worsening execution, and a cancellation that clears
  the best quote within 1.421 seconds. The second fires when an execution itself
  clears the best quote. Both indicate that displayed depth cannot absorb the
  incoming order flow.
- W2 combines a balance-worsening passive liquidity addition, an
  imbalance-worsening cancellation, and an imbalance-worsening execution
  within 21.94 seconds. It captures agreement across liquidity supply,
  withdrawal, and consumption rather than repeated instances of one message
  type.

Both WSELOB rule sets predict a higher rate of 30-second volatility-burst
onsets. They describe predictive order-book configurations and do not claim
that one order causes the later burst.

W1 had F1 p=6.72e-13, maximum F2 p=3.86e-4, F3 lower bound 4.414, and
RW p=3.86e-4. W2 had F1 p=1.44e-6, maximum F2 p=0.00196, F3 lower bound
1.305, and RW p=0.00487.

The fit-sample search sent 13 WSELOB rule sets to certification. Two passed
and the other 11 were rejected. The table below reports every rejected rule
set. F1 tests held-out improvement. Max F2 is the largest direction and
necessity p-value among the rules in the set. The F3 value is the lower
confidence bound on the robust gain across stock-days, and it must be
positive. F1 is a separate 0.05 gate. The F2 rule tests and F3 robustness test
form an intersection test. RW is its Romano--Wolf family-adjusted p-value and
must be below 0.05.

| ID | Candidate rule set | Window | F1 p | Max F2 p | F3 lower bound | RW p | Financial interpretation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| R1 | Restoring liquidity add + imbalance-worsening execution + cancellation that clears the quote; cancellation that clears the quote + immediate replenishment as inhibition; execution that clears the quote | 21.866 s; 0.087 s; 0 | 2.03e-20 | 0.150 | 8.20 | 0.604 | The long-window fragility signal generalized, but the proposed resilience effect from immediate replenishment was weak. The larger rule set therefore did not survive family-wide correction. |
| R2 | Imbalance-worsening liquidity add + cancellation + execution; cancellation that clears the quote during a calm market | 1.382 s; 0 | 0.374 | 0.0126 | -18.1 | 1.000 | Queue clearing in an otherwise calm regime was episodic. The combination was directionally clear for some stock-days but did not improve prediction consistently outside the fit sample. |
| R3 | Imbalance-worsening liquidity add + cancellation + execution | 5.933 s | 0.694 | 0.0374 | -37.1 | 1.000 | This is a shorter version of certified W2. A few seconds of one-sided flow can be absorbed as temporary imbalance, whereas W2 requires persistence over 21.94 seconds. |
| R4 | Inside-spread liquidity add + cancellation that clears the quote; quote clearing + immediate replenishment as inhibition; execution that clears the quote | 6.428 s; 0.087 s; 0 | 0.842 | 0.842 | -6.22e3 | 1.000 | Improving the quote and then replenishing it usually reflects ordinary liquidity provision. Neither the proposed excitation nor inhibition remained stable across stock-days. |
| R5 | Same mechanisms as R4 with a longer inside-spread formation window | 15.949 s; 0.087 s; 0 | 0.845 | 0.836 | -3.99e3 | 1.000 | Extending the window added more unrelated quote updates without producing a stable volatility warning. |
| R6 | Balance-restoring cancellation + balance-worsening cancellation | 0.183 s | 0.759 | 0.0109 | -39.4 | 1.000 | Opposite cancellation directions in a very short interval are consistent with order replacement or routine quote maintenance, not persistent pressure toward a volatility burst. |
| R7 | Balance-restoring execution + cancellation that clears the quote | 2.166 s | 2.54e-4 | 1.67e-5 | -0.0606 | 0.105 | This was the closest rejected candidate. It improved held-out likelihood and had a clear direction, but its robust lower bound was just below zero, so the effect was not sufficiently consistent across stock-days. |
| R8 | Balance-restoring execution + cancellation that clears the quote | 18.592 s | 0.0272 | 0.00804 | -3.79 | 0.967 | The longer window mixed quote-clearing events with executions that restored balance at unrelated times. It passed F1 but lost distributional robustness and family-wide significance. |
| R9 | Imbalance-worsening execution alone | 0 | 0.852 | 0.0857 | -437 | 1.000 | A fill that worsens imbalance is common and is not by itself evidence that displayed depth is exhausted. The certified rules require supporting liquidity-withdrawal or queue-clearing information. |
| R10 | Cancellation that clears the quote + immediate replenishment as excitation; execution that clears the quote | 0.018 s; 0 | 0.556 | 0.0840 | -50.6 | 1.000 | Immediate replenishment is often a sign of resilience rather than additional fragility. Treating it as excitation did not generalize. |
| R11 | Target-blind stressed-market state alone | 0 | 0.841 | 0.159 | -6.47e5 | 1.000 | The broad regime state did not identify which order books would enter a burst. Specific combinations of liquidity supply, withdrawal, and consumption were needed. |

The rejected candidates are economically plausible order-book patterns. Their
rejection does not claim that they can never precede volatility. It shows why
discovery-sample fit alone is insufficient. Short-lived imbalance, routine
quote replenishment, and broad market stress either failed to generalize,
varied too much across stock-days, or did not survive correction across the
13 candidate rule sets.
