# Think-on-Graph trial (backward walks)

A comparison of `coda.inference.tog.ToGBackwardIcd10Agent` with the prompted
ICD-10 agent on the seven English child narratives that the synthetic_narrative
benchmark reads (`../lang_support/data/bn/references/cases_bn_filtered.json`),
each labelled with a CHAMPS cause. For each case, the prompted ICD-10 agent 
proposes up to three CHAMPS causes, an LLM walks the KG from each cause's ICD-10 
code and checks the cause's typical ProbBase findings, and a final call ranks the 
causes with that evidence. With 3 candidate causes, at most 3 hops per walk and 
one finding check per candidate, plus the first opinion and the final ranking, 
this comes to at most 14 LLM calls per case (1 + 3 × (3 + 1) + 1).

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
