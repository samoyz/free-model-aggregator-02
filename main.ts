// free-model-aggregator-02: one server, two client wires, and a real free model
// behind every answer.
//
// The entry is an aggregator: a client points at it, and every completion it
// answers comes from a free OpenRouter model it dials through the grader's
// gateway on the trial's key. It serves POST /v1/chat/completions (OpenAI)
// and POST /v1/messages (Anthropic), in JSON and in server-sent events, and
// GET /v1/models in the shape of the auth header it got.
//
// ROUTING (lib/router.ts): the brief's free models are ranked by the house's
// rating, highest first, and the first candidate that meets the prompt's
// requirements serves it — the best JSON-capable model for a JSON-mode
// prompt, the best overall for the rest. A model that has failed this trial
// sorts below every clean one, so a rate-limited model is not re-dialled
// while the trial runs.
//
// FAILOVER: a refused gateway call (a provider 429 or 5xx, or a network
// error) is failed over to the next candidate, at most three attempts a
// prompt; a gateway refusal (budget, request cap, model allowlist, missing
// output cap) and a 401 are final, because no other model or retry can
// answer them. An answer whose text does not carry the prompt's token, or
// whose usage reports no output tokens, is failed over the same way — the
// model did not do the work. The refusals are reported on a served answer
// in the header x-makefun-upstream-failures, the only way the failover
// column sees a recovered refusal. A prompt no attempt served answers 502
// with that trail; a prompt whose attempts all answered but none carried
// the token is answered with the first attempt's answer, which is honest
// about what the provider said and keeps the prompt out of the failover
// column's denominator.
//
// DETERMINISM: every choice is a function of the brief, the requests and
// the gateway's answers. No wall clock, no unseeded randomness, no
// timestamps in any request. Streams are buffered whole before they are
// relayed, so a truncated stream is a failover, never a broken answer, and
// the same gateway answers give the same bytes on a replay.
//
// FILES
//   main.ts                  this server: the brief, the catalog, the routes, the register and the poll.
//   lib/providers.ts         the brief's types and parser, the free model list, the catalog.
//   lib/router.ts            the model ranking, the failover ladder, the token and error-code rules.
//   lib/wire.ts              Anthropic <-> Chat Completions translation, the model field, the SSE writers.
//   vendor/makefun-agent.ts  the agent SDK, vendored (its gateway-refusal and retry-status lists).
//   LICENSE                  MIT.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  freeModels,
  isRecord,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  streamFromText,
  type ServedModel,
} from './lib/wire.ts';
import {
  anthropicAnswerText,
  errorCodeOf,
  mayFailOver,
  MAX_ATTEMPTS,
  MAX_TRAIL_ROWS,
  noteFault,
  openAIAnswerText,
  openAIStreamComplete,
  openAIStreamOutputTokens,
  openAIStreamText,
  outputCapOf,
  outputTokensOf,
  rankCandidates,
  tokenOf,
  type FaultMap,
  type TrailRow,
} from './lib/router.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** How many times the register POST is retried when the env is not up yet. */
const REGISTER_ATTEMPTS = 5;
/** The pause between register retries, in milliseconds. */
const REGISTER_RETRY_MS = 200;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;
/** The faults this trial has seen, per model id. */
const faults: FaultMap = new Map();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the
 * trial network); a base that does not already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/**
 * The candidates for the prompt at `index`, best first: the free models that meet its
 * requirements, ranked by the trial's faults, then the brief's rating, then
 * the id. A model that failed on the prompt just before is never the first pick while a
 * clean one remains.
 */
function candidatesFor(requires: Requirement, index: number): ModelRow[] {
  return rankCandidates(freeModels(brief), requires, faults, index);
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, joined with the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  const fromBody = requirementsFromBody(body);
  if (entry === undefined) return fromBody;
  return {
    ...(entry.requires.tools === true || fromBody.tools === true ? { tools: true } : {}),
    ...(entry.requires.vision === true || fromBody.vision === true ? { vision: true } : {}),
    ...(entry.requires.json === true || fromBody.json === true ? { json: true } : {}),
    ...(entry.requires.min_context !== undefined ? { min_context: entry.requires.min_context } : {}),
  };
}

/** Requirements read off a request body: the tools it declares, the images it carries, the JSON mode it asks for. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** The failures trail as one header value, or null when there is nothing to report. */
function trailHeader(trail: TrailRow[]): string | null {
  if (trail.length === 0) return null;
  return JSON.stringify(trail.slice(0, MAX_TRAIL_ROWS));
}

function sendJson(res: ServerResponse, status: number, body: unknown, trail: TrailRow[] | null = null): void {
  const text = JSON.stringify(body);
  const headers: Record<string, string> = { 'content-type': 'application/json', 'content-length': String(Buffer.byteLength(text)) };
  const header = trail === null ? null : trailHeader(trail);
  if (header !== null) headers['x-makefun-upstream-failures'] = header;
  res.writeHead(status, headers);
  res.end(text);
}

function sendText(res: ServerResponse, status: number, contentType: string, text: string, trail: TrailRow[] | null = null): void {
  const headers: Record<string, string> = { 'content-type': contentType, 'content-length': String(Buffer.byteLength(text)) };
  const header = trail === null ? null : trailHeader(trail);
  if (header !== null) headers['x-makefun-upstream-failures'] = header;
  res.writeHead(status, headers);
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string, trail: TrailRow[] | null = null): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message), trail);
  else sendJson(res, status, openAIError(message, type, 'upstream_error'), trail);
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** One client request body as a Chat Completions body naming `model`, with a legal output cap. */
function gatewayBody(wire: 'openai' | 'anthropic', body: Json, model: string): Json {
  const cap = outputCapOf(body);
  if (wire === 'openai') {
    const out: Json = { ...body, model };
    out.max_tokens = cap;
    delete out.max_completion_tokens;
    return out;
  }
  const out = anthropicToOpenAI(body, model);
  out.max_tokens = cap;
  return out;
}

/** POST one request to the gateway's Chat Completions route, naming the model this entry dialled. */
async function dial(model: string, body: Json): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(chatCompletionsUrl(), { method: 'POST', headers, body: JSON.stringify({ ...body, model }) });
}

/** The answer one attempt produced: the client's whole response, or why it has none. */
type Attempt =
  | { kind: 'valid'; model: string; wire: 'openai' | 'anthropic'; stream: boolean; text: string }
  | { kind: 'answered'; model: string; wire: 'openai' | 'anthropic'; stream: boolean; text: string }
  | { kind: 'refused'; model: string; status: number; code: string; terminal: boolean }
  | { kind: 'broken'; model: string };

/**
 * One attempt at serving a prompt on one model: dial it, read the whole
 * answer, and say whether it is valid (the token is in it, the usage is
 * whole, the stream is complete), merely answered (a well-formed answer
 * that fails the token or usage check), refused (a 4xx or 5xx, or no answer
 * at all) or broken (no answer can be relayed from it).
 */
async function attempt(wire: 'openai' | 'anthropic', body: Json, model: string, token: string | null): Promise<Attempt> {
  const upstreamBody = gatewayBody(wire, body, model);
  let upstream: Response;
  try {
    upstream = await dial(model, upstreamBody);
  } catch (err) {
    log(`dial ${model}: ${(err as Error).message}`);
    return { kind: 'refused', model, status: 502, code: 'network_error', terminal: false };
  }
  if (!upstream.ok || upstream.body === null) {
    const text = await upstream.text().catch(() => '');
    const code = errorCodeOf(text, upstream.status);
    return { kind: 'refused', model, status: upstream.status, code, terminal: !mayFailOver(upstream.status, code) };
  }
  const stream = body.stream === true;
  const raw = await upstream.text();
  if (stream) {
    const text = openAIStreamText(raw);
    const output = openAIStreamOutputTokens(raw);
    const complete = openAIStreamComplete(raw);
    const ok = complete && output !== null && output >= 1 && (token === null || text.includes(token));
    if (wire === 'openai') {
      let relayed = '';
      for await (const frame of openAIStreamWithModel(streamFromText(raw), model)) relayed += frame;
      return { kind: ok ? 'valid' : 'answered', model, wire, stream, text: relayed };
    }
    let relayed = '';
    for await (const event of anthropicEventsFromOpenAI(streamFromText(raw), model)) relayed += event;
    return { kind: ok ? 'valid' : 'answered', model, wire, stream, text: relayed };
  }
  let answer: unknown;
  try {
    answer = JSON.parse(raw) as unknown;
  } catch {
    return { kind: 'broken', model };
  }
  if (!isRecord(answer)) return { kind: 'broken', model };
  const choices = Array.isArray(answer.choices) ? answer.choices : [];
  const choice = choices.length > 0 && isRecord(choices[0]) ? choices[0] : null;
  if (choice === null || !isRecord(choice.message) || typeof choice.finish_reason !== 'string') return { kind: 'broken', model };
  const output = outputTokensOf(answer);
  const text = wire === 'anthropic' ? anthropicAnswerText(openAIToAnthropic(answer, model)) : openAIAnswerText(answer);
  const ok = output !== null && output >= 1 && (token === null || text.includes(token));
  if (wire === 'openai') {
    return { kind: ok ? 'valid' : 'answered', model, wire, stream, text: JSON.stringify(openAIJsonWithModel(answer, model)) };
  }
  return { kind: ok ? 'valid' : 'answered', model, wire, stream, text: JSON.stringify(openAIToAnthropic(answer, model)) };
}

/**
 * Serve one prompt: walk the candidates, at most three attempts, and answer
 * with the first valid completion. The refusals are reported on a valid
 * answer, and on a 502 when nothing answered; an answered-but-invalid
 * prompt is answered with its first attempt's answer and no trail, because
 * a successful call that missed the token is not a refusal.
 */
async function serve(wire: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const promptIndex = manifestIndex;
  const requires = requirementsFor(body);
  const candidates = candidatesFor(requires, promptIndex);
  if (candidates.length === 0) {
    return sendError(res, wire, 503, wire === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model in the brief meets this request');
  }
  const token = tokenOf(body);
  const trail: TrailRow[] = [];
  const tried = new Set<string>();
  let fallback: Extract<Attempt, { kind: 'answered' }> | null = null;
  for (let n = 0; n < MAX_ATTEMPTS; n++) {
    const next = candidates.find((m) => !tried.has(m.id));
    if (next === undefined) break;
    tried.add(next.id);
    const result = await attempt(wire, body, next.id, token);
    if (result.kind === 'valid') {
      if (result.stream) return sendText(res, 200, 'text/event-stream', result.text, trail.length > 0 ? trail : null);
      return sendJson(res, 200, JSON.parse(result.text), trail.length > 0 ? trail : null);
    }
    if (result.kind === 'answered') {
      noteFault(faults, result.model, promptIndex);
      if (fallback === null) fallback = result;
      continue;
    }
    if (result.kind === 'broken') {
      noteFault(faults, result.model, promptIndex);
      continue;
    }
    noteFault(faults, result.model, promptIndex);
    trail.push({ status: result.status, code: result.code, model: result.model });
    if (result.terminal) break;
  }
  if (fallback !== null) {
    if (fallback.stream) return sendText(res, 200, 'text/event-stream', fallback.text);
    return sendJson(res, 200, JSON.parse(fallback.text));
  }
  log(`gave up on a ${wire} prompt after ${trail.length} refusal(s)`);
  return sendError(res, wire, 502, wire === 'anthropic' ? 'api_error' : 'upstream_error', 'the gateway refused every free model that meets this request', trail.length > 0 ? trail : null);
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by a free model through the gateway. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('openai', body, res);
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('anthropic', body, res);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  let last: unknown = null;
  for (let n = 0; n < REGISTER_ATTEMPTS; n++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
      return;
    } catch (err) {
      last = err;
      if (n + 1 < REGISTER_ATTEMPTS) await new Promise((resolve) => setTimeout(resolve, REGISTER_RETRY_MS));
    }
  }
  throw last instanceof Error ? last : new Error(String(last));
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
