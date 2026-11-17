# ple-baseline: plan–work–verify harness with MCP tools and compaction

A three-role harness built from the `plan-verify` sample, with the MCP
discovery and context-compaction approaches of `custom-openai` ported in.
It calls the Google-sponsored `google-cloud/gemini-3.8-flash` model, so it
needs no API key.

## How it works

`solve(session)` in `serve.py` runs three roles per task, each a separate
conversation on the same model:

1. **Planner** (read-only tools, first ~10% of the time): inspects the
   task's files and replies with a plan — each deliverable with its exact
   path and format, the method, how to check it, and a "Hard constraints"
   section listing every mandatory requirement in the task.
2. **Worker** (all tools, until 25% of the time remains): executes the
   plan. Its prompt makes hard constraints the first priority and requires
   a final one-by-one re-check of each constraint against the actual files
   before finishing. Its answer becomes `final_message`.
3. **Verifier** (read-only tools, fresh conversation with only the task,
   the plan, and the worker's summary): checks the hard constraints first,
   then the rest of the deliverables. It replies `PASS` or a list of gaps;
   gaps go back to the worker for up to 2 fix rounds. Fix rounds wrap up
   `WRAP_UP_SEC` (240 s) before the deadline, so verification can never
   consume the time reserved for final deliverables and the summary.

Cross-cutting features:

- **MCP tools**: at session start the harness connects to the task's MCP
  servers (`taskbridge.MCPClient`), discovers their tools, and offers them
  to the model alongside `bash`/`read_file`/`write_file`. A small catalog
  (≤ 40,000 chars of JSON) is offered directly as `<server>__<tool>`
  functions; a larger one is listed in the system prompt and reached
  through `mcp_find`/`mcp_call`. The planner and verifier only receive
  tools the server annotates `readOnlyHint`, and tool calls outside a
  role's offered set are refused.
- **Compaction**: when a call's prompt size passes `COMPACT_AT_TOKENS`
  (default 150,000), older tool results the model has already read are
  truncated (the 12 most recent are kept); if that is not enough, the
  older turns are replaced by notes the model writes about its own work.
  A context-length 400 triggers the same compaction with `force=True`;
  the base samples' truncate-oldest-half recovery remains as a fallback.
- **Retries**: transient API failures (429, 5xx, dropped connections) are
  retried with backoff without polluting the conversation; each step's
  in-flight call is stopped at its wrap-up threshold and the model is
  asked for its answer with `tool_choice="none"`.
- Token usage is reported after each call via `session.usage`, including
  the compaction-notes calls. Assistant messages are appended via
  `model_dump()` so provider-specific fields (like Gemini's thought
  signatures) are preserved across turns.

| File | Purpose |
|---|---|
| `serve.py` | Agent loop, roles, MCP integration, and compaction |
| `taskbridge.py` | Serve protocol implementation (shared across Python samples) |
| `requirements.txt` | Pinned `openai` and `websockets` dependencies |
| `submission.yaml` | Model declaration, build command, and server startup command |

## Customizing

Because the OpenAI SDK reads `OPENAI_BASE_URL` and `OPENAI_API_KEY` (which
the evaluator sets for every model), you can switch to any sponsored model
just by changing `model_api` in `submission.yaml`.

- For models with smaller context windows, set a lower `MAX_OUTPUT_TOKENS`
  (default 32000) or `COMPACT_AT_TOKENS` under `env:` in `submission.yaml`.
- Always generate your `artifacts/` with the same `model_api` you submit.
