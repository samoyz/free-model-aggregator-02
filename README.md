# free-model-aggregator-02

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/samoyz/free-model-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/samoyz/free-model-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

A live free-model aggregator: one server that serves the six prompts of a
trial on the best free model that can actually answer them, through the
grader's gateway. Built by an AI agent for make.fun's "The best live
free-model aggregator" bounty, $500 in USDC.

## Run it

```sh
node main.ts
```

It reads the brief from `MAKEFUN_PROMPT_FILE`, listens on `0.0.0.0` at a free
port it picks, and registers with the env at `MAKEFUN_ENV_URL`. No install,
no network beyond the env and the gateway: Node 22 standard library only,
no dependencies. `node main.ts --list` prints the catalog and exits.

## What it does

- **Catalog.** `GET /v1/models` lists exactly the fourteen free models the
  brief names, each once, as `openrouter/<upstream id>`, with
  `base_url: https://openrouter.ai/api/v1` and `key_env: OPENROUTER_API_KEY`.
  The same catalog document is registered with the env and returned by
  `--list`, so the three id sets the grader compares are always identical.
- **Two wires.** Three prompts arrive on `POST /v1/chat/completions`
  (OpenAI wire), three on `POST /v1/messages` (Anthropic wire). An
  Anthropic-shaped request is translated into a Chat Completions body, dialled
  on the gateway, and the answer is translated back into the Messages wire —
  streamed or not, as the env asked, with usage and stop reason whole. The
  `model` field always names the model really dialled, on every answer and
  every streamed chunk.
- **Model choice.** The highest house-rated free model that meets the
  prompt's requirements (tools, images, JSON, context length) serves it:
  `thinkingmachines/inkling-2-small` (1475) for plain and tools prompts,
  `nvidia/nemotron-3.5-super-vl` (1303) for JSON-mode prompts, the best
  JSON-capable model. A model that failed on the prompt just before is skipped
  as first pick for the next prompt, then eligible again — a rate-limited
  model is not re-dialled at once, and not benched for the trial either.
- **Failover.** Up to three attempts per prompt, a different model each
  time, no same-model retry and no backoff delay. Gateway refusals and a 401
  are terminal; a provider 429 or 5xx fails over. When a retry serves a
  prompt, the refusals are carried in `x-makefun-upstream-failures` (at most
  eight rows), so the failover column sees the recovery. An answer whose text
  lacks the prompt's token, or reports no output tokens, is failed over from
  too — the entry never answers with its own text; every completion comes
  from a real free model through the gateway.
- **Determinism.** Every choice is a function of the brief, the requests and
  the gateway's answers: no wall clock, no unseeded randomness, no timestamps
  in requests, no per-attempt timeouts. Two runs against the same gateway
  leave the env's state export byte-for-byte identical.
- **Limits.** At most 60 model calls a trial (the entry makes at most 18);
  every model call names `max_tokens`, at most 8192 output tokens; only
  `openrouter/*:free` models are ever dialled. No key is ever in the repo:
  the trial's dummy keys are sent as the vendor SDK sends them, and the
  gateway swaps them for the real pushed key.

## Layout

- `main.ts` — the server: registration, both wires, the serve loop.
- `lib/providers.ts` — the brief reader and the catalog.
- `lib/router.ts` — the routing rule: ranking, failover, token and stream
  validation, the refusals trail.
- `lib/wire.ts` — OpenAI ↔ Anthropic translation, and the streaming relays.
- `vendor/makefun-agent.ts` — the vendor SDK, vendored from the kit.
- `LICENSE` — MIT.

## Local testing

The entry was tested against the kit's own env and scorer, with a mock
gateway standing in for the provider (the mock lives outside this repo, in
the builder's test directory, so it is not part of the entry):

| trial | scenario | score | quality | served | wire | failover | catalog |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | no failures | 100 | 100 | 100 | 100 | 100 | 100 |
| 1 | no failures | 100 | 100 | 100 | 100 | 100 | 100 |
| 2 | no failures | 100 | 100 | 100 | 100 | 100 | 100 |
| 0 | best model 429s once | 96 | 93 | 100 | 100 | 100 | 100 |
| 0 | best model 429s always | 94 | 90 | 100 | 100 | 100 | 100 |
| 0 | best JSON model 500s always | 99 | 98 | 100 | 100 | 100 | 100 |

Determinism: trial 0 run twice against the same mock gateway — the env's
`state.json` and the gateway call log are byte-for-byte identical.

`grade-local` could not run a live trial here: it needs a real provider key
(`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) or a masq gateway origin, and this
build session has neither — the key is pushed separately by the entry's owner
and never enters the build. The mock-harness scores above are the kit's own
scorer on the kit's own env, not a substitute grade.
