# Ground Truth Rubric

The agent must answer with the city name `Paris` on its own line.

Acceptable variations:

- `Paris` (canonical answer)
- `paris` (case-insensitive variant)

Anything else -- including the country (`France`), a longer sentence
that does not isolate `Paris` on its own line, or a refusal -- counts
as a low score on `response_quality`.

This rubric is attached via `llm_scorer_evidence_files` so the LLM
judge sees it but the agent never does.
