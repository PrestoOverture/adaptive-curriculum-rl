# Adaptive Curriculum RL

English | [中文](README.zh-CN.md)

**When you train an AI agent with reinforcement learning, does it learn more if it practises the problem types it
keeps getting wrong, instead of problems drawn at random?**

This is a controlled comparison with the experiment plan written down in advance and the same amount of training for
both groups. The model is Qwen3-4B, trained with GRPO + LoRA on a single RTX 4090.

## Highlights

- **Practising weak spots works: success rate is 3.1 percentage points higher than random practice**
  (95% confidence interval +1.7 to +4.5). The test uses service types never seen in training. All three independent
  training runs (three random seeds) come out ahead, both groups get the same amount of training, and the rule for
  what counts as "better" was fixed before any test data existed.
- **Found a silent bug in the training framework, then redid the whole experiment.** A later audit showed that
  veRL 0.7.0 logged one learning rate and used another, so the first run trained at about 10% of the planned rate.
  The redo checks every step's actual learning rate against the planned curve (to within 1e-12) and uses a fresh
  test set.
- **Results that didn't pan out are reported too.** Plain RL training gives no significant gain over the original
  model. A follow-up was going to test whether the trained model could plan its own practice. A cheap pre-check, run
  before any training, showed that mechanism wasn't working at all, so the follow-up stopped there instead of
  training on something that would only have measured noise.
- **Every number reproduces offline, byte-for-byte**, from the per-attempt test outputs saved in the repository.
  No GPU needed.
- **The trained weights are public** on [Hugging Face](https://huggingface.co/PrestoOverture/octorl-qwen3-4b-grpo-lora),
  each file with a sha256 checksum.

## How the experiment works

**The task.** The agent has to fix a broken service configuration. It has three tools: read the config, look up
information, and submit a fix. It gets at most 5 tool calls per problem. There are three kinds of fault: a config
that breaks a constraint, a missing dependency, and an outdated version. Some problems have nothing wrong with them,
to check that the model doesn't change things that work. Problems come from 10 different services: 6 are used for
training, 2 for tuning, and 2 are held back for the final test. A program checks the fix: 1 point if it is fixed,
0 if not.

**Training.**

1. The original model first trains for 20 rounds. This is the shared starting point for both groups.
2. That starting point is copied into two groups, and each trains for 60 more rounds (in 6 stages, 240 problems and
   960 attempts per group, exactly the same amount of training):
   - **Random practice**: problems always come in the question bank's original mix.
   - **Weak-spot practice**: before each stage, look at which fault type the model got wrong most over the last
     20 rounds, and give it more of that type. No type may exceed 60%, so it never practises only one thing.
3. In each round the model answers the same problem 4 times. Approaches that worked are reinforced and ones that
   failed are discouraged. This is the basic idea of GRPO.
4. The whole procedure is run with three different random seeds, to check that the conclusion holds up.

**Evaluation.** 160 fault problems from the 2 services never seen in training, each answered 4 times, give the
success rate. Another 40 problems with nothing wrong check that the model's pass rate there stays within 5 percentage
points of the original model. Every model answers with identical random seeds, so the comparison is fair.

## Results

| Comparison | Difference in success rate (95% CI) | Verdict |
|---|---|---|
| Weak-spot − random | **+3.07 pp [+1.72, +4.53]** | Better (all three seeds positive) |
| Random − original model | +1.35 pp [+0.10, +2.81] | Not enough evidence (only one seed clearly improves) |

Most of the gain comes from the missing-dependency faults (+9.5 pp). That is the type the weak-spot group practised
most: 42.5–55% of its training problems, against 31% for the random group.

Per-seed numbers, training settings, decision rules and limitations are in [STUDY.md](STUDY.md).

## What's here

| | |
|---|---|
| [STUDY.md](STUDY.md) | Full results, experiment design, safeguards, offline reproduction steps, limitations |
| [ENVIRONMENT.md](ENVIRONMENT.md) | Earlier deliverable: an RL environment for fixing bugs in code repositories (automatic bug injection, Docker sandbox, cheat-resistant grading) |
| `src/tasks/tool_recovery/` | The config-repair task used for training |
| `src/curriculum/` | Weak-spot problem selection, and the code for letting the model plan its own practice |
| `artifacts/self_improve/contracts/` | The experiment plans for each phase, written in advance and locked by hash |

## License

See `LICENSE`.
