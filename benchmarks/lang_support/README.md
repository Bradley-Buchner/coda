# Language support benchmark

Datasets are keyed by `bn` for Bengali, `pt_br` for Brazilian Portuguese,
`ts` for Tsonga, the language also known as Shangaan, `zu` for Zulu, and `st`
for Sesotho. Each contains the same seven recorded COD case narratives.

```
lang_support/
  data/
    bn/
      audio/          # original recording filenames; local, ignored by Git
      references/     # original JSON files, unchanged
    pt_br/
      audio/
      references/
      originals/      # original delivery archive; local, ignored by Git
    ts/
      audio/
      references/
    zu/
      audio/
      references/     # built from the supplied translation document
      originals/      # supplied translation document; local, ignored by Git
    st/
      audio/
      references/
      originals/
  languages/          # adapters for supplied filenames and JSON quirks
  results/
    bn/               # existing results; local, ignored by Git
    pt_br/
```

`dataset_io.load_samples(language)` returns samples with `case_id`, `audio_path`,
and `reference`. The Bengali adapter uses `cases_bn_filtered.json`; the full
`cases_bn.json` is retained as supplied. The Portuguese adapter handles `Iri_*`
filenames and Portuguese text stored under the stale `bn_narrative` field.
The Zulu and Sesotho recordings arrived in one batch with inconsistent
filenames, so their adapters map each supplied filename to its case. Three Zulu
recordings (`lri_1`, `lri_2`, `diarrhea_1`) open with a spoken title that is not
in the reference, which adds a few insertions to their scores.
Duplicate cases, unmatched audio, conflicting references, or missing recordings
produce explicit errors. Place the original recordings under the corresponding
`audio/` directory before running; recordings are not committed to Git.

The existing runners remain available during consolidation. From the repo root:

```bash
PYTHONPATH="$PYTHONPATH:benchmarks/lang_support" python benchmarks/lang_support/run_benchmark.py --language bn indic-conformer
PYTHONPATH="$PYTHONPATH:benchmarks/lang_support" python benchmarks/lang_support/run_benchmark.py --language pt_br whisper-small
PYTHONPATH="$PYTHONPATH:benchmarks/lang_support" python benchmarks/lang_support/run_benchmark.py --language ts mms-1b-all
PYTHONPATH="$PYTHONPATH:benchmarks/lang_support" python benchmarks/lang_support/run_benchmark.py --language zu anv-zul-turbo
PYTHONPATH="$PYTHONPATH:benchmarks/lang_support" python benchmarks/lang_support/run_benchmark.py --language st anv-sot-turbo
PYTHONPATH="$PYTHONPATH:src" python benchmarks/lang_support/bn_asr_bench.py --help
python benchmarks/lang_support/pt_br_asr_bench.py --help
python benchmarks/lang_support/plot_engines.py --results_dir benchmarks/lang_support/results/bn
```

Results under `results/<language>/` were produced in one pass by the current
pipeline, scoring both languages over the same seven cases. Each file records the
host hardware, the faster-whisper compute type, and any clips that failed.

Engines that reach a rate-limited API are retried, and a clip that never returns
text is recorded under `failed` rather than scored as a complete
mis-transcription. Only the successful attempt is timed, so a retry does not
inflate RTF.

`indic-whisper` is a Hindi fine-tune kept as a cross-script control. It
recognizes Bengali speech but writes it in Devanagari, so its word error rate
exceeds 1.0 by design and should not be read as a broken engine.
