# Adaptive Curriculum RL — full study write-up

Detailed results, design, safeguards and reproduction steps behind the [README](README.md) summary.
The project was originally called OctoRL; code identifiers, file paths and artifacts keep that name.
The README's "weak-spot practice" is the **failure-driven** arm here, and "random practice" is the **fixed** arm.

Does practising more on what an agent currently fails at make RL post-training better, when the budget is
held equal? And can a model that has been trained this way design its own curriculum, as a first RSI-related test?
This study answers both questions with preregistered, budget-matched experiments. The model is Qwen3-4B, trained with
GRPO + LoRA on a single RTX 4090, and every result is judged on sealed held-out tests. Negative and null results are
reported as they came out.

## Results

Primary endpoint: **test2**, 160 held-out fault instances × 4 rollouts from service families never seen in
training, plus 40 normal instances as a guardrail. Fault-recovery success rate (%):

| Model | seed 42 | seed 137 | seed 2718 |
|---|---:|---:|---:|
| Base Qwen3-4B | 61.25 | 61.25 | 61.25 |
| Fixed-distribution GRPO | 61.09 | 60.63 | 66.09 |
| Failure-driven GRPO | 63.59 | 64.38 | 69.06 |

| Question | Per-seed difference (pp) | Pooled, 95% CI (pp) | Preregistered verdict |
|---|---|---|---|
| **Q2**: failure-driven − fixed | +2.50 / +3.75 / +2.97 | **+3.07 [+1.72, +4.53]** | `Q2_failure_driven_better` |
| **Q1**: fixed − base | −0.16 / −0.63 / +4.84 | +1.35 [+0.10, +2.81] | `no_evidence_of_improvement` (not all seeds > 0) |
| **Q4 (R6)**: does training make the model a better curriculum designer? | — | chooser TV 0.003 vs. threshold 0.10 | `Q4_mechanism_not_exercised` (stopped before training) |

- The Q2 gain is concentrated in one fault type, missing-dependency (+9.5 pp on 52 instances). This is the type the
  failure-driven selector up-weighted (42.5–55% of training rows, against 31% under the fixed distribution).
- CIs are instance-cluster bootstraps. They cover test-instance and evaluation-sampling uncertainty only; the
  training-seed variance is **not** estimated (n = 3).
- R6 is a bounded negative measurement. On identical inputs, the trained checkpoint chooses practice mixes that are
  indistinguishable from the base model's (the difference is below sampling noise). Task training did not transfer
  into curriculum design, so the recursive mechanism was never exercised. This says nothing about RSI in general.
- The six R4r adapters behind this table are on Hugging Face:
  [PrestoOverture/adaptive-curriculum-rl-qwen3-4b-grpo-lora](https://huggingface.co/PrestoOverture/adaptive-curriculum-rl-qwen3-4b-grpo-lora).

## Study design

**Environment** (`src/tasks/tool_recovery/`). A multi-turn tool-use task in which the agent repairs a service
configuration. It has three tools (`read_config`, `query_info`, `submit_fix`) and at most 5 tool calls per episode.
There are three fault types (constraint violation, missing dependency, stale version) plus normal instances, and ten
service families, split 6 train / 2 dev / 2 test. Reward is binary and computed by a verifier. Every instance is
seed-regenerable, and content fingerprints guarantee zero overlap between the train, dev and test sets. The environment
is a pure-Python state machine; the Agent-R1 adapter is `src/adapters/tool_recovery_agent_r1.py`.

**Training.** Qwen3-4B (revision `1cfa9a7`) with veRL 0.7.0, Agent-R1 and vLLM 0.10.2. GRPO uses a sign advantage
(A = 2r − 1), 4 groups × 4 rollouts per update, and temperature 1.5. LoRA is r = 16 / α = 32, and the learning rate is
2e-5 on a cosine schedule with a 100-step horizon.

**Comparison.** Each seed (42, 137, 2718) forks both arms from a shared 20-update checkpoint. Each arm then trains a
further 60 updates in 6 stages, which is 240 task groups / 960 rollouts per arm; the budget is matched in raw groups.
The fixed arm samples the pool distribution π0. The failure-driven arm re-weights each stage toward failing fault types
with `p_c = 0.4·π0_c + 0.6·π0_c·f_c / Σ_k π0_k·f_k`, where `f_c` is the smoothed failure rate over the previous 20
updates and every type is capped at 60%.

**Decision rules**, frozen before any test data existed. A comparison is "better" only if all three per-seed
differences are > 0, the pooled 95% CI lower bound is > 0, and the normal-instance pass rate stays within 5 pp of base.
Evaluation uses identical per-(instance, rollout) sampling seeds for every model.

## How the study protected itself

- **Preregistration with hash locks.** Each phase has a frozen contract in `artifacts/self_improve/contracts/`.
  Amendments are dated and state which data had been seen when they were made. The run queues refuse to start if the
  contract's hash changes.
- **Audit-caught protocol deviation → full re-run.** An offline audit found that veRL 0.7.0 silently overwrote the
  learning-rate horizon in staged training (the logged config is printed *before* the overwrite). The original R4 run
  therefore trained at about 10% of the planned cumulative learning rate. R4 was re-run as **R4r** with code-enforced gates:
  every logged LR matched a reference curve within 1e-12, and task selection was byte-identical to the frozen selector.
  R4r was evaluated on a freshly generated test set, because R4's test had already been opened. R4's verdicts are kept only
  as a record of the deviation.
- **Futility gates before training.** R6 checked, before any training, whether the mechanism it wanted to test was
  actually present. It was not, so the phase stopped before running training that would have measured noise.
- **Synthetic controls for every statistic.** These include A/A runs giving CI [0, 0], null coverage of 0.95, and an
  inversion test that caught a spec error in which the R6 input-sensitivity formula ran in the wrong direction.

## Reproduce the numbers (offline, no GPU)

The per-rollout evaluation outputs (`artifacts/self_improve/r4r/test_eval/`) are in the repository. Both commands below
reproduce the frozen analysis and gate files byte-for-byte; this was verified on 2026-09-30.

```bash
uv sync --extra dev
for endpoint in test2 test_r3; do
  extra=(); [ "$endpoint" = test_r3 ] && extra=(--secondary)
  args=()
  for m in base fixed_42 fixed_137 fixed_2718 failure_driven_42 failure_driven_137 failure_driven_2718; do
    args+=(--model "$m=artifacts/self_improve/r4r/test_eval/$endpoint/$m.json")
  done
  uv run scripts/self_improve/r4r_analysis.py "${args[@]}" "${extra[@]}" \
    --out-json /tmp/r4r_$endpoint.json --out-md /tmp/r4r_$endpoint.md
  cmp /tmp/r4r_$endpoint.json artifacts/self_improve/r4r/r4r_analysis_$endpoint.json && echo "$endpoint reproduced"
done

# R6 futility gate from the saved chooser completions
uv run scripts/self_improve/r6_offline_replay.py --phase g2 \
  --completions artifacts/self_improve/r6/replay_real/g1/completions.jsonl --output-dir /tmp/r6_g2
cmp /tmp/r6_g2/g2_metrics.json artifacts/self_improve/r6/replay_real/g2/final14/g2_metrics.json && echo "G2 reproduced"
```

The report verifier is
`uv run scripts/self_improve/r5_verify_report.py`. LoRA weights are not in Git. The six R4r U80 adapters are published on
[Hugging Face](https://huggingface.co/PrestoOverture/adaptive-curriculum-rl-qwen3-4b-grpo-lora), byte-identical to the manifest (see
`artifacts/self_improve/hf_release/RELEASE.md`). All 20 adapters are indexed with
sha256 values in `artifacts/self_improve/r5/checkpoint_manifest.json`. Local weight files were removed after the release;
`artifacts/self_improve/pruned_model_files.json` records each removed file's sha256 and where it is archived.

## Where things are

| Path | Contents |
|---|---|
| `artifacts/self_improve/contracts/` | Frozen preregistrations (R3, R4, R4r, R6) |
| `artifacts/self_improve/r4r/RESULTS.md` | Primary Q1/Q2 results and diagnostics |
| `artifacts/self_improve/r5/R5_report.md` | Provenance, training diagnostics, limitations, adapter manifest |
| `artifacts/self_improve/r6/R6_report.md` | Model-as-curriculum-designer extension and its futility gate |
| `src/tasks/tool_recovery/` | The environment: generator, state machine, verifier, records |
| `src/curriculum/` | Failure-driven selector (`failure_driven.py`) and model chooser (`model_chooser.py`) |
| `scripts/self_improve/` | Stage runners, gates, evaluation, analysis (`r3*` … `r6*`) |

Some older artifacts link to `docs/*.md`. Those are the author's private design notes and are not published.

## Limitations

- n = 3 training seeds. Seed-to-seed variance is visible: on Q1, seed 2718 is +4.8 pp while the other two seeds are about 0.
- One synthetic task family, one 4B model, one short training budget (80 updates). The effect is modest and
  concentrated in one fault type.
- The failure-driven result is evidence about *this selection rule under this budget*. It is not evidence of general self-improvement.
- An exploratory gap is unexplained: the R3c U80 checkpoint scores 64.7% versus 60.6% for the fixed arm of the same
  seed. The learning rate is ruled out; the candidates are row order, per-stage restarts and seed noise.
