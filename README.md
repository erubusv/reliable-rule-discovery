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

Generated configurations and outputs are stored under
`runs/reproduction/seed-<id>/`. The reference data digests for run 111 are:

- Aave: `5f790a169904e6e6c77db7010e9a3aef858276c643fcf3b4fff0969bfa8c6ea7`
- WSELOB: `287a6e4c015b5280cf19d260c9fa0430e0d94cc299b9257ec01397a13f634974`

Per-seed reports are written to each run directory. Their mean and sample
standard deviation are written to
`runs/reproduction/metrics_three_seeds.json`.

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

## Anonymous release

The public release must be created from a fresh root commit. Removing a name
from the latest files or adding it to `.gitignore` does not remove it from old
commits. Do not publish development history, local paper files, data, logs, or
results. Use an account and repository that are not linked to an author, and
verify the public commit metadata before sharing the URL.

Create a one-commit release repository with:

```bash
tools/create_anonymous_release.sh /path/to/anonymous-release
```

The new repository contains only publishable files allowed by `.gitignore`. It
has no remote and no connection to the development history. Push that directory
to a new anonymous repository rather than force-pushing the development remote.

## Validation

```bash
pytest
```

The regression suite checks likelihood calculations, high-order completion,
history-count rules, dependency-aware model selection, route decisions,
certification, baseline preparation, and metric reporting.
