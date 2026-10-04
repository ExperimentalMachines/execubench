# ExecuServe benchmark contract, version 1

What the phone-side server must provide before any P1 device minute is spent. The harness
(`execubench/harness/run.py`) calls `GET /v1/execuserve/capabilities` first and refuses to run
unless the response contains `"benchmark": {"contract": 1, "features": [...]}` with every
feature below. ExecuServe today (2026-10-04) uses the ExecuTorch 1.4.0 AAR and implements none
of this; the work belongs in the ExecuServe repository, against this document.

## Runtime

- **ExecuTorch 1.5.1** (`org.pytorch:executorch-android:1.5.1`, the version in
  `config/versions.env`). Maven has no 1.5.0. In 1.5.1, `LlmModuleConfig`'s `dataPath` default
  changed from `""` to `null`; check ExecuServe's module construction against it.
- The benchmark build is its own flavour with a pinned APK sha256 (`runtime.apk_sha256` in the
  pilot manifest), reported by the capabilities endpoint as `benchmark.apk_sha256` and recorded
  in every job. The harness refuses to run when the manifest pins no hash or the server reports
  another.

## Why the stock 1.5.1 Android API is not enough

Checked in the v1.5.1 sources (tag commit `3b60683923245cf472b7323426920e15623ba361`):

1. **`maxNewTokens` is not forwarded.** `LlmModule.generate(prompt, config, callback)` passes
   `config.seqLen`, echo, temperature and the BOS/EOS counts to JNI, never `maxNewTokens`, and
   the JNI builds its native `GenerationConfig` from `seq_len` alone
   (`LlmModule.kt` line 378, `jni_layer_llama.cpp` line 284). Setting the builder field does not
   cap output. 1.4.0 behaves the same.
2. **`seqLen` is not a safe substitute for these exports.** Every v1 export has
   `get_max_seq_len` 2,048 (the prefill chunk) and `get_max_context_len` 8,192 (read from the 16
   export reports). With `max_seq_len < max_context_len` the 1.5.1 text runner takes its
   sliding-window branch: it checks only `num_prompt_tokens < max_context_len` and resolves the
   generation limit with zero occupied positions (`text_llm_runner.cpp`, the `effective_pos`
   lines). With a KV cache reused across turns, nothing in the runtime stops a conversation
   from running past the 8k cache, and "Max seq length exceeded" is never raised.
3. **Text callbacks are not tokens.** The JNI buffers token pieces until they form valid UTF-8
   before calling `onResult`, so one callback can carry several sampled tokens. Counting
   callbacks does not count tokens, and the time of the first callback is not the time of the
   first sampled token.
4. **No `ignore_eos`.** The native config has it; the Android `LlmGenerationConfig` does not
   (`numEos` is about prompt encoding, not stopping).
5. **Prefill-only then generate.** With an empty prompt after a separate prefill, the runner
   reports `prompt_tokens` as the accumulated position, not the timed suffix.
6. **The runner's clock is `CLOCK_REALTIME`** (`util.h`, `time_in_ms`), integer milliseconds.

## Features

| Feature | Request | Response, in `x_execuserve` |
|---|---|---|
| `cache_off` | `"x_execuserve": {"cache": "off"}` resets the runner before the request; `"on"` keeps exact-prefix reuse | `cached_tokens` (0 when off) |
| `native_max_new_tokens` | `"max_tokens": N` is enforced natively: a patched JNI forwards `max_new_tokens`, so the runner stops after N sampled tokens | `sampled_tokens` as counted natively |
| `capacity_guard` | Always on | Refuses with HTTP 400 `context_overflow` when occupied positions + prompt tokens + `max_tokens` exceed `get_max_context_len`, tracked by the server itself because the runtime does not (point 2) |
| `raw_runner_stats` | Always on | `runner_stats`: the runner's stats JSON, verbatim and unparsed |
| `native_monotonic` | Always on | `monotonic`: `call_start_ns`, `first_sampled_token_ns` (taken in the native token callback before UTF-8 buffering), `first_callback_ns`, `call_end_ns`, all from `CLOCK_MONOTONIC`/`elapsedRealtimeNanos` |
| `prompt_echo` | Always on | `prompt_text`: the exact rendered prompt bytes the runner received |
| `effective_threads` | Always on | `effective_threads`: the XNNPACK thread count after load |
| `status_memory` | `GET /v1/execuserve/status` | `/proc/self/status` `VmRSS`, `VmHWM`, `RssAnon`, `RssFile` in kB, for phones where the adb shell cannot read another app's `/proc` |

The first six are required (`REQUIRED_FEATURES` in `execubench/harness/run.py`; a test keeps
the two lists equal); `effective_threads` and `status_memory` are required before P3.

## Acceptance tests

Run on a phone (P1 experiment `contract_acceptance`) and on the JVM dev server where possible.
Each is a request, the expected outcome, and what is checked.

| Test | Request | Pass when |
|---|---|---|
| One-token cap | `max_tokens: 1` | `generated_tokens` 0, `sampled_tokens` 1, one output token |
| Ordinary cap | `max_tokens: 64` on a prompt that does not stop early | `generated_tokens` 63, `sampled_tokens` 64 |
| End of sequence | A prompt that stops before the cap | `finish_reason: "stop"`, `generated_tokens` counts the EOS step |
| Cache off | Same prompt twice, `cache: "off"` | `cached_tokens` 0 both times; `prompt_tokens` equal |
| Cache on | Two turns of one conversation, `cache: "on"` | turn 2 `cached_tokens` > 0 and `prompt_tokens` equals the new suffix only |
| Near the window | Occupied + prompt + cap = window - 1, then = window + 1 | First succeeds; second is a 400 `context_overflow` before any runner call |
| Split UTF-8 | A prompt whose reply starts with a multi-byte character split across tokens | `first_sampled_token_ns` < `first_callback_ns`; text is valid UTF-8 |
| Empty output | `max_tokens: 1` where the first token is special | `runner_stats` present, `prompt_text` present, no crash |
| Clock cross-check | Any request | runner TTFT and decode within 5 ms + 1 percent of the native monotonic durations (`docs/METRICS.md`) |
| Prompt echo | Any templated request | sha256 of `prompt_text` equals sha256 of `/apply-template` output |

A build passes contract 1 when every test passes on one phone of each SoC vendor in Tier A.
