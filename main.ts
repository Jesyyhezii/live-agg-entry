// live-agg-entry: one server, two client wires, real free models behind every answer.
//
// The entry is an aggregator for free-model-aggregator-02. A client points at it as if it were
// OpenAI (POST /v1/chat/completions) or Anthropic (POST /v1/messages); every completion it answers
// comes from a real free OpenRouter model dialled through the grader's gateway on Chat Completions.
// It never answers from its own weights: there are none here.
//
// How a trial runs:
//   1. the server listens on 0.0.0.0 at a port the OS picks and registers it, with its catalog,
//      at POST $MAKEFUN_ENV_URL/register;
//   2. it answers GET /v1/models in the shape the auth header names (Bearer -> OpenAI,
//      x-api-key -> Anthropic) and the six prompts one at a time, JSON or SSE as asked;
//   3. each prompt names model "auto": the server replaces it with the best free model the brief
//      rates for that prompt's requirements, dials it on the gateway, and relays the answer with
//      its usage and stop reason whole, the `model` field always the id it really dialled;
//   4. a refused gateway call (4xx/5xx) is retried on the next best free model that meets the
//      prompt, and the refusals are reported in the x-makefun-upstream-failures header;
//   5. it polls GET $MAKEFUN_ENV_URL/work and exits 0 the moment the env says done.
//
// Determinism: nothing here reads the wall clock or unseeded randomness. Every choice is a
// function of the brief, the requests and the gateway's answers, so a replay leaves the same
// bytes. Fixed timestamps stand in wherever a reply needs a `created` value.
//
// Run as `node main.ts` (Node 22+, standard library only) or `node main.ts --list` to print the
// catalog document and exit 0. No key is ever read from or written to this repository: the trial's
// dummy key comes from OPENAI_API_KEY in the environment and the gateway swaps it for the
// entrant's own pushed OpenRouter key.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** The wire protocol's version, carried in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The gateway refuses a call with no output cap, or one over this many tokens. */
const MAX_OUTPUT_TOKENS = 8192;
/** The output cap a request gets when it names none. */
const DEFAULT_MAX_TOKENS = 1024;
/** The most model calls one trial may spend; past it the gateway refuses with 429. */
const MAX_GATEWAY_CALLS = 60;
/** One gateway dial may take this long before it counts as refused. */
const DIAL_TIMEOUT_MS = 25_000;
/** A fixed epoch second wherever a reply needs a `created` value. */
const FIXED_CREATED = 1767225600;
/** A fixed instant wherever a reply needs a `created_at` value. */
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_ROOT = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key; the gateway swaps it for the entrant's own pushed key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

type Json = Record<string, unknown>;
const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);
const asString = (v: unknown): string => (typeof v === 'string' ? v : '');
const asNumber = (v: unknown): number => (typeof v === 'number' && Number.isFinite(v) ? v : 0);

function log(message: string): void {
  process.stderr.write(`live-agg: ${message}\n`);
}

/** The text of a string, or of a block array's text blocks joined with newlines. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

/** A rough token count for a usage estimate: about one token per four characters, at least one. */
function estimateTokens(text: string): number {
  return Math.max(1, Math.ceil(text.length / 4));
}

// ---------------------------------------------------------------------------
// The brief: the free model list and the six prompts' manifest
// ---------------------------------------------------------------------------

interface ModelRow {
  id: string;
  provider: string;
  displayName: string;
  ratingMilli: number;
  contextLength: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface Requirement {
  tools: boolean;
  vision: boolean;
  json: boolean;
}

interface PromptRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: number;
  requires: Requirement;
}

interface Brief {
  models: ModelRow[];
  prompts: PromptRow[];
  error: string | null;
}

/**
 * The fourteen free models, as the brief lists them, baked in so `--list` can print the full
 * catalog even outside a trial, where no brief file exists. Inside a trial the brief is the
 * source of truth and this table is only the fallback when the brief cannot be read; the two
 * agree, row for row, on the practice instance.
 */
const FALLBACK_MODELS: ModelRow[] = [
  { id: 'openrouter/baidu/ernie-5.5-flash-preview:free', provider: 'openrouter', displayName: 'Baidu: ERNIE-5.5-Flash-Preview (free)', ratingMilli: 1150, contextLength: 1024000, tools: true, vision: false, json: true, free: true },
  { id: 'openrouter/cohere/north-mini-code:free', provider: 'openrouter', displayName: 'Cohere: North Mini Code (free)', ratingMilli: 1200, contextLength: 256000, tools: true, vision: false, json: false, free: true },
  { id: 'openrouter/dots-studio/dots-3-note-preview:free', provider: 'openrouter', displayName: 'Dots Studio: Dots3-Note Preview (free)', ratingMilli: 1150, contextLength: 512000, tools: true, vision: true, json: true, free: true },
  { id: 'openrouter/google/gemma-4-26b-a4b-it:free', provider: 'openrouter', displayName: 'Google: Gemma 4 26B A4B (free)', ratingMilli: 1150, contextLength: 262144, tools: true, vision: true, json: true, free: true },
  { id: 'openrouter/google/gemma-4-31b-it:free', provider: 'openrouter', displayName: 'Google: Gemma 4 31B (free)', ratingMilli: 1210, contextLength: 262144, tools: true, vision: true, json: true, free: true },
  { id: 'openrouter/inclusionai/ling-3.2-flash:free', provider: 'openrouter', displayName: 'inclusionAI: Ling 3.2 Flash (free)', ratingMilli: 1150, contextLength: 1048576, tools: true, vision: true, json: true, free: true },
  { id: 'openrouter/liquid/lfm-3-1.7b:free', provider: 'openrouter', displayName: 'LiquidAI: LFM3-1.7B (free)', ratingMilli: 1150, contextLength: 262144, tools: true, vision: false, json: true, free: true },
  { id: 'openrouter/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free', provider: 'openrouter', displayName: 'NVIDIA: Nemotron 3 Nano Omni (free)', ratingMilli: 1150, contextLength: 256000, tools: true, vision: true, json: false, free: true },
  { id: 'openrouter/nvidia/nemotron-3-ultra-550b-a55b:free', provider: 'openrouter', displayName: 'NVIDIA: Nemotron 3 Ultra (free)', ratingMilli: 1285, contextLength: 1000000, tools: true, vision: false, json: false, free: true },
  { id: 'openrouter/nvidia/nemotron-3.5-lightning:free', provider: 'openrouter', displayName: 'NVIDIA: Nemotron 3.5 Lightning (free)', ratingMilli: 1221, contextLength: 1000000, tools: true, vision: false, json: false, free: true },
  { id: 'openrouter/nvidia/nemotron-3.5-super-vl:free', provider: 'openrouter', displayName: 'NVIDIA: Nemotron 3.5 Super VL (free)', ratingMilli: 1303, contextLength: 1000000, tools: true, vision: true, json: true, free: true },
  { id: 'openrouter/particle-ai/particle-1:free', provider: 'openrouter', displayName: 'Particle AI: Particle-1 (free)', ratingMilli: 1150, contextLength: 1048576, tools: true, vision: true, json: true, free: true },
  { id: 'openrouter/recursive/flint-v1.0:free', provider: 'openrouter', displayName: 'Recursive: Flint (free)', ratingMilli: 1150, contextLength: 262144, tools: true, vision: false, json: true, free: true },
  { id: 'openrouter/thinkingmachines/inkling-2-small:free', provider: 'openrouter', displayName: 'Thinking Machines: Inkling 2 Small (free)', ratingMilli: 1475, contextLength: 1048576, tools: true, vision: true, json: false, free: true },
];

/** The index of the `}` closing the object that opens at `start`, or -1; strings are skipped. */
function matchingBrace(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in `text` carrying a string `schema`, in the order they appear. */
function schemaDocuments(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = matchingBrace(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not a document on its own: look on.
    }
  }
  return out;
}

function modelRowOf(v: unknown): ModelRow | null {
  if (!isRecord(v)) return null;
  const raw = asString(v.id) !== '' ? asString(v.id) : asString(v.model);
  if (raw === '') return null;
  const provider = asString(v.provider) !== '' ? asString(v.provider) : 'openrouter';
  const id = provider === 'openrouter' && !raw.toLowerCase().startsWith('openrouter/') ? `openrouter/${raw}` : raw;
  return {
    id,
    provider,
    displayName: asString(v.display_name) !== '' ? asString(v.display_name) : raw,
    ratingMilli: typeof v.rating_milli === 'number' ? v.rating_milli : 0,
    contextLength: typeof v.context_length === 'number' ? v.context_length : null,
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free !== false,
  };
}

function promptRowOf(v: unknown): PromptRow | null {
  if (!isRecord(v)) return null;
  const id = asString(v.id);
  if (id === '') return null;
  const req = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: { tools: req.tools === true, vision: req.vision === true, json: req.json === true },
  };
}

/** The brief at `path`: the free model list and the prompts' manifest, found by schema. */
function readBrief(path: string): Brief {
  if (path === '') return { models: [], prompts: [], error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { models: [], prompts: [], error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = schemaDocuments(text).filter((d) => asString(d.schema).includes('free-model-aggregator-02/'));
  const modelsDoc = docs.find((d) => Array.isArray(d.models) || Array.isArray(d.ratings));
  if (modelsDoc === undefined) {
    return { models: [], prompts: [], error: `the brief at ${path} holds no models document` };
  }
  const rows = (Array.isArray(modelsDoc.models) ? modelsDoc.models : modelsDoc.ratings) as unknown[];
  const models = rows.flatMap((v) => {
    const m = modelRowOf(v);
    return m === null ? [] : [m];
  });
  const promptsDoc = docs.find((d) => Array.isArray(d.prompts) || Array.isArray(d.manifest));
  const promptRows = promptsDoc === undefined ? [] : ((Array.isArray(promptsDoc.prompts) ? promptsDoc.prompts : promptsDoc.manifest) as unknown[]);
  const prompts = promptRows.flatMap((v) => {
    const p = promptRowOf(v);
    return p === null ? [] : [p];
  });
  return { models, prompts, error: null };
}

const brief = readBrief(PROMPT_FILE);
/** The free models this entry may dial: the brief's, else the baked-in table. */
const MODELS: ModelRow[] = brief.models.length > 0 ? brief.models : FALLBACK_MODELS;

// ---------------------------------------------------------------------------
// The catalog
// ---------------------------------------------------------------------------

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface Catalog {
  providers: Array<{
    id: string;
    base_url: string;
    key_env: string;
    wire: 'openai';
    models: CatalogModel[];
  }>;
}

/**
 * The one catalog document: one provider, exactly the free models this entry may dial, once
 * each, in brief order. The base_url is OpenRouter's public endpoint, never the gateway's
 * trial-local address, so the document is identical on a live pass and its replay.
 */
function buildCatalog(): Catalog {
  const seen = new Set<string>();
  const models: CatalogModel[] = [];
  for (const model of MODELS) {
    if (!model.free || seen.has(model.id)) continue;
    seen.add(model.id);
    models.push({ id: model.id, free: true, context_length: model.contextLength });
  }
  if (models.length === 0) return { providers: [] };
  return {
    providers: [
      {
        id: 'openrouter',
        base_url: 'https://openrouter.ai/api/v1',
        key_env: 'OPENROUTER_API_KEY',
        wire: 'openai',
        models,
      },
    ],
  };
}

const CATALOG = buildCatalog();
const CATALOG_TEXT = `${JSON.stringify(CATALOG)}\n`;

// ---------------------------------------------------------------------------
// Routing: the best free model for a prompt, and its failover order
// ---------------------------------------------------------------------------

/** Whether a model can serve a prompt with these requirements. */
function meets(model: ModelRow, req: Requirement): boolean {
  if (req.tools && !model.tools) return false;
  if (req.vision && !model.vision) return false;
  if (req.json && !model.json) return false;
  return true;
}

/**
 * The free models that can serve `req`, best first by the brief's rating_milli. The sort is
 * stable, so equal ratings keep the brief's order. Quality is 60% of the score, so the highest
 * rated eligible model goes first and every later row is its failover, in the same order.
 */
function candidatesFor(req: Requirement): ModelRow[] {
  return MODELS.filter((m) => m.free && meets(m, req)).sort((a, b) => b.ratingMilli - a.ratingMilli);
}

/** Which prompt row the request now arriving is: the manifest is in the client's arrival order. */
let manifestIndex = 0;

/** Requirements read off a request body, when the brief holds no row for it. */
function requirementsFromBody(body: Json, wire: 'openai' | 'anthropic'): Requirement {
  let vision = false;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

function requirementsFor(body: Json, wire: 'openai' | 'anthropic'): Requirement {
  const row = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (row !== undefined) return row.requires;
  return requirementsFromBody(body, wire);
}

// ---------------------------------------------------------------------------
// The gateway: one Chat Completions dial, with failover
// ---------------------------------------------------------------------------

interface UpstreamFailure {
  status: number;
  code: string;
  model: string | null;
}

/** How many gateway calls this trial has spent; the gateway refuses past 60. */
let gatewayCalls = 0;

function chatCompletionsUrl(): string {
  if (GATEWAY_ROOT === '') return '';
  return GATEWAY_ROOT.endsWith('/v1') ? `${GATEWAY_ROOT}/chat/completions` : `${GATEWAY_ROOT}/v1/chat/completions`;
}

/** A refusal's code, from the gateway's error body: its code, its type, else its HTTP status. */
function failureCodeOf(text: string, status: number): string {
  try {
    const parsed: unknown = JSON.parse(text);
    if (isRecord(parsed) && isRecord(parsed.error)) {
      const code = asString(parsed.error.code);
      if (code !== '') return code.slice(0, 80);
      const type = asString(parsed.error.type);
      if (type !== '') return type.slice(0, 80);
    }
  } catch {
    // Not JSON: fall through to the status.
  }
  return `http_${status}`;
}

interface DialSuccess {
  ok: true;
  model: string;
  response: Response;
  failures: UpstreamFailure[];
}

interface DialFailure {
  ok: false;
  failures: UpstreamFailure[];
}

/**
 * Dial `body` on the gateway, trying the best eligible model first and failing over down the
 * rating order, one call per model, until one answers or the list (or the 60-call cap) runs
 * out. Every refusal is recorded for the x-makefun-upstream-failures header.
 */
async function dialWithFailover(body: Json, req: Requirement): Promise<DialSuccess | DialFailure> {
  const failures: UpstreamFailure[] = [];
  for (const model of candidatesFor(req)) {
    if (gatewayCalls >= MAX_GATEWAY_CALLS) break;
    gatewayCalls += 1;
    let upstream: Response;
    try {
      upstream = await fetch(chatCompletionsUrl(), {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          authorization: `Bearer ${GATEWAY_KEY}`,
          ...(body.stream === true ? { accept: 'text/event-stream' } : {}),
        },
        body: JSON.stringify({ ...body, model: model.id }),
        signal: AbortSignal.timeout(DIAL_TIMEOUT_MS),
      });
    } catch {
      failures.push({ status: 0, code: 'network_error', model: model.id });
      continue;
    }
    if (upstream.ok && upstream.body !== null) return { ok: true, model: model.id, response: upstream, failures };
    const text = await upstream.text().catch(() => '');
    failures.push({ status: upstream.status, code: failureCodeOf(text, upstream.status), model: model.id });
  }
  return { ok: false, failures };
}

/** The failures header: at most 8 rows, in the order the calls refused. */
function failuresHeader(failures: UpstreamFailure[]): Record<string, string> {
  if (failures.length === 0) return {};
  return { 'x-makefun-upstream-failures': JSON.stringify(failures.slice(0, 8)) };
}

// ---------------------------------------------------------------------------
// Wire shapes: errors, model lists, translations
// ---------------------------------------------------------------------------

function openAIError(message: string, type: string): Json {
  return { error: { message, type, code: null } };
}

function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** GET /v1/models on a Bearer request: OpenAI's list shape. */
function openAIModelList(): Json {
  return {
    object: 'list',
    data: MODELS.filter((m) => m.free).map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an x-api-key request: Anthropic's list shape. */
function anthropicModelList(): Json {
  return {
    data: MODELS.filter((m) => m.free).map((m) => ({ type: 'model', id: m.id, display_name: m.displayName, created_at: FIXED_CREATED_AT })),
    has_more: false,
  };
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
function anthropicStopOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** A tool call's arguments as an object: the gateway's JSON string, or {} when it is not one. */
function argumentsObject(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null otherwise. */
function openAIPartOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds other things. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/**
 * An Anthropic Messages request as a Chat Completions request: the system prompt becomes a system
 * message, image blocks become image_url data URLs, tool_use blocks become tool_calls, tool_result
 * blocks become tool messages, tools and tool_choice are rewritten, and the output cap,
 * temperature, top_p, stop sequences and streaming pass through.
 */
function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: asString(block.id) !== '' ? asString(block.id) : '',
          type: 'function',
          function: {
            name: asString(block.name),
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({ role: 'assistant', content: text !== '' ? text : null, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({ role: 'tool', tool_call_id: asString(block.tool_use_id), content: toolResultText(block.content) });
        continue;
      }
      const part = openAIPartOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  out.max_tokens = capTokens(body.max_tokens);
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || asString(raw.name) === '') return [];
      return [
        {
          type: 'function',
          function: {
            name: asString(raw.name),
            description: asString(raw.description),
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && asString(choice.name) !== '') out.tool_choice = { type: 'function', function: { name: asString(choice.name) } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** The output cap a request carries, clamped to the gateway's limit, defaulted when absent. */
function capTokens(value: unknown): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) return DEFAULT_MAX_TOKENS;
  return Math.min(Math.max(1, Math.floor(value)), MAX_OUTPUT_TOKENS);
}

/** An OpenAI-shaped client body readied for the gateway: the cap always named, never over 8192. */
function openAIRequestBody(body: Json, model: string): Json {
  const out: Json = { ...body, model };
  if (typeof out.max_completion_tokens === 'number') out.max_completion_tokens = capTokens(out.max_completion_tokens);
  else out.max_tokens = capTokens(out.max_tokens);
  if (out.stream === true) out.stream_options = { include_usage: true };
  return out;
}

/**
 * A Chat Completions answer as an Anthropic Messages object, `model` the id this entry dialled.
 * Usage comes from the gateway whole; when it is absent, an estimate keeps the wire's counts
 * well-formed (input at least 0, output at least 1).
 */
function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: asString(call.id) !== '' ? asString(call.id) : 'toolu_unknown',
      name: asString(call.function.name),
      input: argumentsObject(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  const output = typeof usage.completion_tokens === 'number' ? usage.completion_tokens : estimateTokens(text);
  return {
    id: `msg_${asString(body.id) !== '' ? asString(body.id) : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: anthropicStopOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: asNumber(usage.prompt_tokens), output_tokens: Math.max(1, output) },
  };
}

/** A Chat Completions answer with `model` the id this entry dialled, usage kept well-formed. */
function openAIJsonAnswer(body: Json, model: string): Json {
  const out: Json = { ...body, model };
  if (!isRecord(out.usage)) {
    let text = '';
    if (Array.isArray(out.choices) && isRecord(out.choices[0]) && isRecord(out.choices[0].message)) {
      text = textOf(out.choices[0].message.content);
    }
    const output = estimateTokens(text);
    out.usage = { prompt_tokens: 0, completion_tokens: output, total_tokens: output };
  }
  return out;
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split a byte stream into SSE frames on the blank line that ends each one. */
async function* sseFrames(stream: ReadableStream<Uint8Array>): AsyncGenerator<string> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        yield buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') yield buffer;
  } finally {
    reader.releaseLock();
  }
}

/** The data payloads of one SSE frame, parsed when they are JSON objects. */
function framePayloads(frame: string): Array<{ line: string; payload: Json | null; raw: string }> {
  const out: Array<{ line: string; payload: Json | null; raw: string }> = [];
  for (const line of frame.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed.startsWith('data:')) continue;
    const raw = trimmed.slice(5).trim();
    if (raw === '' || raw === '[DONE]') {
      out.push({ line, payload: null, raw });
      continue;
    }
    try {
      const parsed: unknown = JSON.parse(raw);
      out.push({ line, payload: isRecord(parsed) ? parsed : null, raw });
    } catch {
      out.push({ line, payload: null, raw });
    }
  }
  return out;
}

/**
 * A Chat Completions SSE stream relayed as it came, every data payload's `model` the id this
 * entry dialled. When the gateway sends no usage chunk, a final one is appended (estimated from
 * the text relayed) so the wire's usage counts stay well-formed, before `data: [DONE]`.
 */
async function* openAIStreamRelay(stream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let sawUsage = false;
  let sawDone = false;
  let textLength = 0;
  for await (const frame of sseFrames(stream)) {
    const payloads = framePayloads(frame);
    for (const { raw } of payloads) if (raw === '[DONE]') sawDone = true;
    const lines = frame.split('\n').map((line) => {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) return line;
      const raw = trimmed.slice(5).trim();
      if (raw === '' || raw === '[DONE]') return line;
      try {
        const chunk: unknown = JSON.parse(raw);
        if (isRecord(chunk)) {
          if (isRecord(chunk.usage)) sawUsage = true;
          const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
          const delta = isRecord(choices[0]) ? choices[0] : null;
          if (delta !== null && isRecord(delta.delta) && typeof delta.delta.content === 'string') textLength += delta.delta.content.length;
          return `data: ${JSON.stringify({ ...chunk, model })}`;
        }
      } catch {
        // Not JSON: relay the line as it came.
      }
      return line;
    });
    yield `${lines.join('\n')}\n\n`;
  }
  if (!sawUsage) {
    const output = estimateTokens(' '.repeat(textLength));
    yield `data: ${JSON.stringify({ id: 'chatcmpl-aggregate', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [], usage: { prompt_tokens: 0, completion_tokens: output, total_tokens: output } })}\n\n`;
  }
  if (!sawDone) yield 'data: [DONE]\n\n';
}

/**
 * A Chat Completions SSE stream translated into the Anthropic event sequence: message_start on
 * the first chunk, content_block_start/delta/stop per text block or tool call, then message_delta
 * with the stop reason and usage, then message_stop. `model` is the id this entry dialled.
 */
async function* anthropicStreamRelay(stream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;
  let sawUsage = false;
  let textLength = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeBlocks = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    }
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (): string[] => {
    if (finished) return [];
    finished = true;
    if (!sawUsage) outputTokens = estimateTokens(' '.repeat(textLength));
    return [
      ...closeBlocks(),
      sseEvent('message_delta', {
        type: 'message_delta',
        delta: { stop_reason: pendingStop ?? 'end_turn', stop_sequence: null },
        usage: { output_tokens: Math.max(1, outputTokens) },
      }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  for await (const frame of sseFrames(stream)) {
    for (const { payload, raw } of framePayloads(frame)) {
      if (raw === '[DONE]') {
        yield* startEvents();
        yield* finishEvents();
        continue;
      }
      if (payload === null) continue;
      if (isRecord(payload.usage)) {
        sawUsage = true;
        inputTokens = asNumber(payload.usage.prompt_tokens);
        outputTokens = asNumber(payload.usage.completion_tokens);
      }
      const choices = Array.isArray(payload.choices) ? payload.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      yield* startEvents();
      if (typeof delta.content === 'string' && delta.content !== '') {
        textLength += delta.content.length;
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          yield sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } });
        }
        yield sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } });
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            yield sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex });
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          yield sseEvent('content_block_start', {
            type: 'content_block_start',
            index,
            content_block: {
              type: 'tool_use',
              id: asString(call.id) !== '' ? asString(call.id) : `toolu_${index}`,
              name: asString(fn.name),
              input: {},
            },
          });
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          yield sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } });
        }
      }
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = anthropicStopOf(choice.finish_reason);
    }
  }
  if (!finished) {
    yield* startEvents();
    yield* finishEvents();
  }
}

// ---------------------------------------------------------------------------
// HTTP plumbing
// ---------------------------------------------------------------------------

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

function sendJson(res: ServerResponse, status: number, body: unknown, extraHeaders: Record<string, string> = {}): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...extraHeaders });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string, extraHeaders: Record<string, string> = {}): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message), extraHeaders);
  else sendJson(res, status, openAIError(message, type), extraHeaders);
}

/** A refused or broken gateway call as a 502 in the client's wire: the prompt is lost. */
function refused(res: ServerResponse, wire: 'openai' | 'anthropic', failures: UpstreamFailure[]): void {
  const last = failures[failures.length - 1];
  const detail = last !== undefined ? `the gateway answered HTTP ${last.status} (${last.code}) on ${last.model ?? 'no model'}` : 'the gateway answered no model';
  sendError(res, wire, 502, wire === 'anthropic' ? 'api_error' : 'upstream_error', detail, failuresHeader(failures));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function modelsRoute(req: IncomingMessage, res: ServerResponse): void {
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList());
  else sendJson(res, 200, openAIModelList());
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by a free model. */
async function chatRoute(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const dialed = await dialWithFailover(openAIRequestBody(body, 'auto'), requirementsFor(body, 'openai'));
  if (!dialed.ok) return refused(res, 'openai', dialed.failures);
  const headers = failuresHeader(dialed.failures);
  if (body.stream === true) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...headers });
    try {
      for await (const frame of openAIStreamRelay(dialed.response.body, dialed.model)) res.write(frame);
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = await dialed.response.json().catch(() => null);
  if (!isRecord(answer)) return sendError(res, 'openai', 502, 'upstream_error', 'the gateway answered no JSON object', headers);
  sendJson(res, 200, openAIJsonAnswer(answer, dialed.model), headers);
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messagesRoute(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const dialed = await dialWithFailover(anthropicToOpenAI(body, 'auto'), requirementsFor(body, 'anthropic'));
  if (!dialed.ok) return refused(res, 'anthropic', dialed.failures);
  const headers = failuresHeader(dialed.failures);
  if (body.stream === true) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...headers });
    try {
      for await (const event of anthropicStreamRelay(dialed.response.body, dialed.model)) res.write(event);
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = await dialed.response.json().catch(() => null);
  if (!isRecord(answer)) return sendError(res, 'anthropic', 502, 'api_error', 'the gateway answered no JSON object', headers);
  sendJson(res, 200, openAIToAnthropic(answer, dialed.model), headers);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    modelsRoute(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chatRoute(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messagesRoute(req, res);
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

// ---------------------------------------------------------------------------
// Register, poll, exit
// ---------------------------------------------------------------------------

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** POST /register with the port and the one catalog; the env answers with the prompt count. */
async function register(port: number): Promise<void> {
  let lastError = '';
  for (let attempt = 0; attempt < 10; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: CATALOG }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
      return;
    } catch (err) {
      lastError = (err as Error).message;
      await sleep(200);
    }
  }
  throw new Error(`could not register with the env: ${lastError}`);
}

/** Poll GET /work; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status === 200) {
          const work: unknown = await res.json();
          if (isRecord(work) && work.done === true) {
            clearInterval(timer);
            server.close();
            process.exit(0);
          }
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte. Wait for stdout to flush.
    await new Promise<void>((resolve) => process.stdout.write(CATALOG_TEXT, () => resolve()));
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
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${MODELS.filter((m) => m.free).length} free models`);
  await register(port);
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
