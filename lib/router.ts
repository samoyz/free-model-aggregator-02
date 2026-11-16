// The routing rule: which free model serves a prompt, in what order the entry
// tries them, and what a failed attempt leaves behind.
//
// Quality is 60% of the score and it is a ratio against the best free model that
// meets the prompt, so the first candidate is always the highest-rated model
// that meets the requirements. The rest of the order is the failover ladder:
// every candidate that meets the requirements, highest rating first, ties by
// id, so two runs that see the same brief and the same gateway answers make
// the same choices.
//
// A model that failed on the immediately preceding prompt sorts below every
// model that did not: a rate-limited model is not re-dialled at once, but it
// is not benched for the trial either, because a 429 that cleared in the
// meantime is worth the best rating again, and the most contended model is
// the one every entry dials first. The ladder is walked top to bottom, at
// most three attempts a prompt, and a model is never tried twice on one
// prompt.
//
// Nothing here reads a clock or a random source: the order is a function of
// the brief, the requests and the gateway's answers, so a replay that serves
// the same answers makes the same calls.

import { GATEWAY_REFUSALS, RETRY_STATUSES } from '../vendor/makefun-agent.ts';
import { capabilitiesOf, meetsRequirements, type Json, type ModelRow, type Requirement } from './providers.ts';

/** One row of the optional `x-makefun-upstream-failures` trail. */
export interface TrailRow {
  status: number;
  code: string;
  model: string | null;
}

/** The most attempts one prompt costs, whatever fails. */
export const MAX_ATTEMPTS = 3;
/** The most rows the failures trail may carry (the env reads at most 8). */
export const MAX_TRAIL_ROWS = 8;
/** The most output tokens a gateway call may ask for (the gateway refuses more). */
export const MAX_OUTPUT_TOKENS = 8192;
/** The output cap a gateway call names when the client named none. */
export const DEFAULT_OUTPUT_TOKENS = 1024;
/** The token a prompt's instruction carries, e.g. ZEBRA-4721. */
export const TOKEN_RE = /[A-Z]+-\d{4}/;

/** The faults this trial has seen: per model id, the index of the prompt it last failed on. */
export type FaultMap = Map<string, number>;

/** Whether a model is cooling down for the prompt at `index`: it failed on the prompt just before. */
export function isCooling(faults: FaultMap, model: string, index: number): boolean {
  return faults.get(model) === index - 1;
}

/** Record a fault against a model, on the prompt at `index`. */
export function noteFault(faults: FaultMap, model: string, index: number): void {
  faults.set(model, index);
}

/**
 * The candidates for the prompt at `index`, best first: the free models that
 * meet the requirements, models that did not fail on the prompt just before
 * ahead of the ones that did, then highest rating, then lowest id. The order
 * is total, so the walk is deterministic.
 */
export function rankCandidates(models: readonly ModelRow[], requires: Requirement, faults: FaultMap, index: number): ModelRow[] {
  const candidates = models.filter((m) => m.free && meetsRequirements(requires, capabilitiesOf(m)));
  candidates.sort((a, b) => {
    const ca = isCooling(faults, a.id, index) ? 1 : 0;
    const cb = isCooling(faults, b.id, index) ? 1 : 0;
    if (ca !== cb) return ca - cb;
    if (a.rating_milli !== b.rating_milli) return b.rating_milli - a.rating_milli;
    return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
  });
  return candidates;
}

/** The unique token a request body carries, or null when it names none. */
export function tokenOf(body: Json): string | null {
  const found = TOKEN_RE.exec(JSON.stringify(body));
  return found === null ? null : found[0];
}

/** The text an OpenAI completion answer carries (content, else tool-call arguments). */
export function openAIAnswerText(answer: Json): string {
  const choices = Array.isArray(answer.choices) ? answer.choices : [];
  const choice = choices.length > 0 && typeof choices[0] === 'object' && choices[0] !== null ? (choices[0] as Json) : {};
  const message = typeof choice.message === 'object' && choice.message !== null ? (choice.message as Json) : {};
  const parts: string[] = [];
  if (typeof message.content === 'string') parts.push(message.content);
  if (Array.isArray(message.content)) {
    for (const block of message.content) {
      if (typeof block === 'object' && block !== null && (block as Json).type === 'text' && typeof (block as Json).text === 'string') {
        parts.push((block as Json).text as string);
      }
    }
  }
  if (Array.isArray(message.tool_calls)) {
    for (const call of message.tool_calls) {
      if (typeof call === 'object' && call !== null) {
        const fn = (call as Json).function;
        if (typeof fn === 'object' && fn !== null && typeof (fn as Json).arguments === 'string') parts.push((fn as Json).arguments as string);
      }
    }
  }
  return parts.join('\n');
}

/** The text an Anthropic message answer carries. */
export function anthropicAnswerText(answer: Json): string {
  const content = Array.isArray(answer.content) ? answer.content : [];
  const parts: string[] = [];
  for (const block of content) {
    if (typeof block === 'object' && block !== null && (block as Json).type === 'text' && typeof (block as Json).text === 'string') {
      parts.push((block as Json).text as string);
    }
  }
  return parts.join('');
}

/** The text of a buffered OpenAI SSE body: every chunk's delta content, in order. */
export function openAIStreamText(sse: string): string {
  let text = '';
  for (const frame of sse.split('\n\n')) {
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '' || payload === '[DONE]') continue;
      try {
        const chunk: unknown = JSON.parse(payload);
        if (typeof chunk !== 'object' || chunk === null || Array.isArray(chunk)) continue;
        const choices = (chunk as Json).choices;
        if (!Array.isArray(choices) || choices.length === 0) continue;
        const choice = choices[0];
        if (typeof choice !== 'object' || choice === null) continue;
        const delta = (choice as Json).delta;
        if (typeof delta === 'object' && delta !== null && typeof (delta as Json).content === 'string') text += (delta as Json).content as string;
      } catch {
        // A frame that is not JSON carries no text.
      }
    }
  }
  return text;
}

/** Whether a buffered OpenAI SSE body is a complete stream: a finish chunk and [DONE]. */
export function openAIStreamComplete(sse: string): boolean {
  let sawFinish = false;
  let sawDone = false;
  for (const frame of sse.split('\n\n')) {
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        sawDone = true;
        continue;
      }
      if (payload === '') continue;
      try {
        const chunk: unknown = JSON.parse(payload);
        if (typeof chunk !== 'object' || chunk === null || Array.isArray(chunk)) continue;
        const choices = (chunk as Json).choices;
        if (!Array.isArray(choices) || choices.length === 0) continue;
        const choice = choices[0];
        if (typeof choice === 'object' && choice !== null && typeof (choice as Json).finish_reason === 'string') sawFinish = true;
      } catch {
        // Not JSON: not a finish chunk.
      }
    }
  }
  return sawFinish && sawDone;
}

/** The output-token count a buffered OpenAI SSE body's usage chunks report (the last one), or null when none does. */
export function openAIStreamOutputTokens(sse: string): number | null {
  let output: number | null = null;
  for (const frame of sse.split('\n\n')) {
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '' || payload === '[DONE]') continue;
      try {
        const chunk: unknown = JSON.parse(payload);
        if (typeof chunk !== 'object' || chunk === null || Array.isArray(chunk)) continue;
        const usage = (chunk as Json).usage;
        if (typeof usage !== 'object' || usage === null || Array.isArray(usage)) continue;
        const out = (usage as Json).completion_tokens;
        if (typeof out === 'number' && Number.isFinite(out)) output = out;
      } catch {
        // Not JSON: no usage here.
      }
    }
  }
  return output;
}

/** The output-token count a Chat Completions answer reports, or null when it reports none. */
export function outputTokensOf(answer: Json): number | null {
  const usage = typeof answer.usage === 'object' && answer.usage !== null ? (answer.usage as Json) : null;
  if (usage === null) return null;
  const out = usage.completion_tokens;
  return typeof out === 'number' && Number.isFinite(out) ? out : null;
}

/**
 * The error code of a gateway refusal body, as the env reads it: the error's
 * `code`, else its `type`, else a code from the status. Capped at 64 chars,
 * as the env caps it.
 */
export function errorCodeOf(text: string, status: number): string {
  try {
    const value: unknown = JSON.parse(text);
    if (typeof value === 'object' && value !== null) {
      const error = (value as Json).error;
      if (typeof error === 'object' && error !== null) {
        const err = error as Json;
        if (typeof err.code === 'string' && err.code !== '') return err.code.slice(0, 64);
        if (typeof err.code === 'number') return `http_${err.code}`.slice(0, 64);
        if (typeof err.type === 'string' && err.type !== '') return err.type.slice(0, 64);
      }
    }
  } catch {
    // Not JSON: fall through to the status.
  }
  return `http_${status}`;
}

/**
 * Whether the code is one of the gateway's own refusals: final, never
 * retried. The SDK names four; the other three refusals are a fault of the
 * request itself (too large, not JSON, stateful), so retrying one against
 * another model would only repeat the same refusal.
 */
export function isGatewayRefusal(code: string): boolean {
  return (GATEWAY_REFUSALS as readonly string[]).includes(code)
    || code === 'request_too_large' || code === 'invalid_json' || code === 'stateful_request';
}

/** Whether the status is one the vendor SDK retries (a provider 429 or 5xx among them). */
export function isRetryStatus(status: number): boolean {
  return (RETRY_STATUSES as readonly number[]).includes(status);
}

/**
 * Whether a refused call may be failed over from: any 4xx or 5xx that is not
 * a gateway refusal and not an authentication failure. A 401 means the trial
 * has no key, so no model can serve; a gateway refusal is final by the
 * gateway's own rule. Everything else is the provider's, and another free
 * model may answer.
 */
export function mayFailOver(status: number, code: string): boolean {
  if (status === 401) return false;
  if (isGatewayRefusal(code)) return false;
  return status >= 400 && status <= 599;
}

/**
 * A gateway call's output cap, made legal: the client's cap when it names one
 * at or under the limit, the default when it names none, the limit when it
 * names more. The cap is a function of the request alone, so a replay builds
 * the same body.
 */
export function outputCapOf(body: Json): number {
  const named = [body.max_tokens, body.max_completion_tokens].filter(
    (v): v is number => typeof v === 'number' && Number.isSafeInteger(v) && v >= 1,
  );
  if (named.length === 0) return DEFAULT_OUTPUT_TOKENS;
  return Math.min(Math.max(...named), MAX_OUTPUT_TOKENS);
}
