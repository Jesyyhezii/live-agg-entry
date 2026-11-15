# live-agg-entry — The best live free-model aggregator

Entry for make.fun `free-model-aggregator-02`. One Node server (`main.ts`, standard
library only) that serves the OpenAI Chat Completions API and the Anthropic Messages
API, in JSON and in streaming, and answers every prompt with a real completion from a
free OpenRouter model dialled through the grader's gateway.

- `POST /v1/chat/completions` and `POST /v1/messages`, JSON + SSE
- `GET /v1/models` in the wire its auth header names (Bearer → OpenAI, x-api-key →
  Anthropic)
- `node main.ts --list` prints the catalog document it registers, byte for byte
- Model choice: the brief's highest `rating_milli` free model that meets the prompt's
  requirements (JSON mode, tools, vision); on a refused gateway call it fails over to
  the next best eligible model and reports the refusals in
  `x-makefun-upstream-failures`
- Registers with the env, polls `GET /work`, exits 0 when the trial is done

Run: `node main.ts` (needs `MAKEFUN_ENV_URL`, `MAKEFUN_PROMPT_FILE`,
`OPENAI_BASE_URL`, `OPENAI_API_KEY` from the trial sandbox).

No API key is stored in this repository. The trial's dummy key comes from the
environment; the gateway swaps it for the entrant's own pushed OpenRouter key.
