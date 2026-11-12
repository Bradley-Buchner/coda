# Think-on-Graph trial (backward walks)

A comparison of `coda.inference.tog.ToGBackwardIcd10Agent` with the prompted
ICD-10 agent on the seven English child narratives that the synthetic_narrative
benchmark reads (`../lang_support/data/bn/references/cases_bn_filtered.json`),
each labelled with a CHAMPS cause. For each case, the prompted ICD-10 agent 
proposes up to three CHAMPS causes, an LLM walks the KG from each cause's ICD-10 
code and checks the cause's typical ProbBase findings, and a final call ranks the 
causes with that evidence. With 3 candidate causes, at most 3 hops per walk and 
one finding check per case, plus the first opinion and the final ranking, 
this comes to at most 12 LLM calls per case (1 + 3 × 3 + 1 + 1).

The ToG agent's first opinion is the prompted agent's answer, so a closing
table lists both top causes against the label from the same run. Seven
synthetic cases can't show whether the KG helps; this is simply a smoke test.

The agent reads the KG from the repository's `kg/` TSV files.

## Run

From the repository root (needs API key):

```bash
python benchmarks/tog_trial/run_trial.py --cases lri_1,malaria_1
python benchmarks/tog_trial/run_trial.py
```

`--provider` and `--model` override the configured inference LLM. Each case is
summarized on screen and saved to `results/<case_id>.json`, with the result
and the prompt and response of every LLM call. `results/` is gitignored.

## Adoption benchmark

Tests whether ToG is worth adopting on `cases.json`, 80 synthetic narratives
across 16 causes. Seven of them were used to tune the prompts, so they are only
a pilot; `chain_cases.json` lists them, the cases where a terminal illness follows
an underlying one, labels awaiting a check, and close causes for lenient scoring.
`cases_sparse.json` holds the other 73 with half of their sentences removed, and
`make_sparse.py` recreates it.

Commit your changes first, then run the pilot, score it, and run the rest:

```bash
python benchmarks/tog_trial/run_benchmark.py --sets pilot
python benchmarks/tog_trial/score_benchmark.py
python benchmarks/tog_trial/run_benchmark.py --sets original,sparse
python benchmarks/tog_trial/make_review_packet.py
```

The pilot makes about 120 LLM calls and the rest about 2,200. Rerunning skips
saved results and retries failed ones. Results go to `results/benchmark/`, and a
run with other code or another model needs a new `--out`.

The arms share each case's first opinion and differ only in the evidence given to
the final call: none, findings only, facts only, all of it, or a placebo where the
candidates' evidence is swapped. The placebo's facts and findings can still name
their original cause, and the second look's prompt reads as though the KG had
nothing to offer.

The `findings_look` arm runs `ToGFindingsIcd10Agent`, which skips the walks: a
new finding check and a final call with a findings-only prompt, 2 calls on top of
the first opinion. To add it to a finished run, commit, then run:

```bash
python benchmarks/tog_trial/run_benchmark.py --sets pilot,original,sparse --arms findings_look
```

`--arms` runs only the arms named, with their repeats, on the saved full results,
and records their code and model in `arm_runs.json`. The arm's repeats rerun its
finding check as well as its final call, so its noise is not measured the same
way as Full ToG's, and its timed seconds leave out the first opinion.

Fill in the CSV files in `results/benchmark/review/`, then run
`make_review_packet.py --score`, or rate them in the terminal with
`rate_packet.py compare`, `evidence` and `findings`. The comparison isn't blind:
each output is labeled, and ToG's reasoning leaves out the sentences code adds
about probability changes.

| Sheet | Column | Values |
|---|---|---|
| `1_ratings.csv` | `more_useful` | A, B or same |
| `1_ratings.csv` | `wrong_or_unsupported_in_A`, `_in_B` | yes or no |
| `2_evidence_check.csv` | `rating` (cited items) | useful, filler or misleading |
| `2_evidence_check.csv` | `adds_beyond_narrative` (reasoning) | yes or no |
| `3_finding_checks.csv` | `correct_answer` | yes, no, not mentioned or unclear |

Answer the finding checks from the narrative alone. Blank cells count as unrated,
and other values stop the scoring so you can correct them.
