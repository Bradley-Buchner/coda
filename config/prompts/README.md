# Prompt Configs

YAML files that define the LLM prompts used by the extractor, the reranker and
the questionnaire answerer. Each config is referenced by **key** (its filename
stem) from the main config, via `grounder.rag.extractor.prompt`,
`grounder.rag.reranker.prompt` and `inference.questionnaire.prompt` in
`config/settings.yaml`.

## Fields

| Field | Required | Description |
|-------|----------|-------------|
| `use_schema` | yes | If `true`, calls the LLM with structured output (JSON schema). If `false`, parses raw JSON from the response text. |
| `system_prompt` | no | System prompt. Supports `{concept_type}` interpolation (extractor only). |
| `user_prompt` | yes | User prompt. Extractor supports `{concept_type}` and `{text}`; reranker supports `{concept}`, `{evidence_text}`, and `{retrieved_terms}`; questionnaire answerer supports `{dialogue}` and `{questions}`. |
| `concept_key` | extractor only | The key for the concept name in the LLM's output. |
| `supporting_evidence_key` | extractor only | The key for the evidence list in the LLM's output. |
| `schema` | if `use_schema: true` | JSON schema for structured output. Not used by the questionnaire answerer, which builds its schema in code (one entry per question, restricted to that question's allowed answers). |

## Output field keys (extractor)

Because different prompts use different field names in their output,
`concept_key` and `supporting_evidence_key` tell the extractor which keys to
read:

```yaml
concept_key: Disease                       # key for the concept name in LLM output
supporting_evidence_key: Supporting Evidence  # key for the evidence list
```

## Provided configs

| File | Description |
|------|-------------|
| `extractor_default.yaml` | General-purpose extractor. Parametric on `concept_type`. Uses structured output. |
| `extractor_medcoder.yaml` | Verbatim prompt from the MedCodER paper for ICD-10 disease extraction. Uses raw JSON output (`use_schema: false`). |
| `reranker_default.yaml` | General-purpose reranker. Uses structured output. |
| `questionnaire_answerer_default.yaml` | Answers retrieved verbal autopsy questions from the interview transcript with a verbatim evidence quote per answer. Uses structured output (schema built in code). |

## Custom configs

Create a new YAML file in this directory following the structure of an existing
config, then reference it by key (its filename stem) from `config/settings.yaml`:

```yaml
# config/prompts/my_extractor.yaml  ->  referenced as `my_extractor`
grounder:
  rag:
    extractor:
      prompt: my_extractor
```
