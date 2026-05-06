import Anthropic from '@anthropic-ai/sdk';
import type { RawMessageStreamEvent } from '@anthropic-ai/sdk/resources/messages/messages';
import type { VoyageAIClient } from 'voyageai';
import type {
  LLMMessage,
  LLMProvider,
  LLMStreamEvent,
  StreamOptions,
} from '@inferagraph/core';

/**
 * A single embedding vector — a flat array of floats.
 *
 * Mirrors the locked `Vector` shape in `@inferagraph/core` so this provider
 * compiles even before core publishes the embeddings extension. Once core
 * exports the type the structural shape is identical.
 */
export type Vector = number[];

/**
 * Per-call options for {@link LLMProvider.embed}.
 *
 * Mirrors the locked `EmbedOptions` contract from `@inferagraph/core`.
 *
 * `signal` is typed via `StreamOptions['signal']` so this package's tsconfig
 * does not need to pull in the DOM lib for the `AbortSignal` global.
 */
export interface EmbedOptions {
  /** Override the provider's default embedding model for this call. */
  model?: string;
  /** Optional cancellation signal; forwarded to the Voyage SDK. */
  signal?: StreamOptions['signal'];
}

/**
 * Optional Voyage AI configuration. When supplied, the returned provider
 * exposes {@link LLMProvider.embed}; when omitted, `embed` is undefined and
 * the provider is chat-only (Tier 1 path — InferaGraph still works without
 * embeddings, just no semantic-search features).
 *
 * Anthropic does not have a native embeddings endpoint. Voyage AI is
 * Anthropic's officially recommended embedding partner — see
 * https://platform.claude.com/docs/en/build-with-claude/embeddings.
 */
export interface VoyageConfig {
  /** Voyage API key. Get one from https://www.voyageai.com/. */
  apiKey: string;
  /**
   * Default Voyage embedding model. `'voyage-3.5'` is the recommended general
   * model; `'voyage-3-large'` is higher quality at higher cost;
   * `'voyage-code-3'` is tuned for code. Default `'voyage-3.5'`.
   */
  model?: string;
  /** Optional Voyage base URL override. Ignored when {@link client} is set. */
  baseURL?: string;
  /**
   * Pre-built Voyage client. When supplied, {@link apiKey} and
   * {@link baseURL} are ignored — the caller is fully responsible for
   * client configuration. Primary use case is testing with a mock.
   */
  client?: VoyageAIClient;
}

/**
 * Configuration for the Anthropic provider.
 *
 * The provider talks to the Anthropic Messages API for chat. Embeddings are
 * served via Voyage AI when {@link voyage} is supplied; otherwise embeddings
 * are unavailable and `embed` is undefined on the returned provider.
 */
export interface AnthropicProviderConfig {
  /**
   * Anthropic API key. NEVER expose this to the browser; pass it from a
   * server-side environment variable.
   *
   * Ignored when {@link client} is provided.
   */
  apiKey: string;
  /** Chat model. Default `'claude-sonnet-4-20250514'`. */
  model?: string;
  /**
   * Optional base URL override. Ignored when {@link client} is provided.
   */
  baseURL?: string;
  /**
   * Pre-built Anthropic client. When supplied, {@link apiKey} and
   * {@link baseURL} are ignored — the caller is fully responsible for
   * client configuration. Primary use case is testing with a mock.
   */
  client?: Anthropic;
  /**
   * Optional Voyage AI integration for embeddings. When set, the returned
   * provider exposes `embed()`; when omitted, `embed` is undefined.
   */
  voyage?: VoyageConfig;
}

/**
 * Default chat model — Claude Sonnet 4.
 */
const DEFAULT_MODEL = 'claude-sonnet-4-20250514';

/**
 * Anthropic requires `max_tokens` on every request. We pick a reasonable
 * fallback so callers don't have to think about it.
 */
const DEFAULT_MAX_TOKENS = 4096;

/**
 * Default Voyage embedding model.
 */
const DEFAULT_VOYAGE_MODEL = 'voyage-3.5';

/**
 * JSON-mode prompt suffix. Anthropic has no `response_format` parameter; the
 * canonical workaround is to instruct the model in-prompt. This is best-effort
 * and the consumer must still validate the output.
 */
const JSON_INSTRUCTION = '\n\nRespond with valid JSON only, no prose.';

/**
 * Translate Anthropic's `stop_reason` values to the contract's narrower set.
 * Anything that isn't a clean `length` falls through to `stop` — `tool_use`
 * and `end_turn` both map to `stop`, which matches AIEngine's expectations.
 */
function translateStopReason(
  reason: string | null | undefined,
): 'stop' | 'length' | 'aborted' {
  if (reason === 'max_tokens') return 'length';
  return 'stop';
}

/**
 * Construct an {@link LLMProvider} backed by Anthropic Claude (chat) and
 * optionally Voyage AI (embeddings).
 *
 * ```ts
 * import { anthropicProvider } from '@inferagraph/anthropic-provider';
 *
 * <InferaGraph
 *   data={data}
 *   llm={anthropicProvider({
 *     apiKey: process.env.ANTHROPIC_API_KEY!,
 *     voyage: { apiKey: process.env.VOYAGE_API_KEY! },
 *   })}
 * />
 * ```
 *
 * When `voyage` is omitted the returned provider has `embed === undefined`;
 * chat still works, but embedding-dependent features (semantic search,
 * similarity highlight) are unavailable.
 */
export function anthropicProvider(config: AnthropicProviderConfig): LLMProvider {
  const client =
    config.client ??
    new Anthropic({
      apiKey: config.apiKey,
      ...(config.baseURL ? { baseURL: config.baseURL } : {}),
    });
  const model = config.model ?? DEFAULT_MODEL;

  const voyageConfig = config.voyage;
  // Voyage client is constructed lazily so tests (and chat-only consumers)
  // never trigger the voyageai package's ESM entry. Tests inject a client
  // directly via `voyage.client`, sidestepping the dynamic import entirely.
  let voyageClientPromise: Promise<VoyageAIClient> | undefined;
  if (voyageConfig) {
    voyageClientPromise = voyageConfig.client
      ? Promise.resolve(voyageConfig.client)
      : import('voyageai').then(
          (mod) =>
            new mod.VoyageAIClient({
              apiKey: voyageConfig.apiKey,
              ...(voyageConfig.baseURL
                ? { baseUrl: voyageConfig.baseURL }
                : {}),
            }),
        );
  }
  const voyageModel = config.voyage?.model ?? DEFAULT_VOYAGE_MODEL;

  const embedFn = voyageClientPromise
    ? async (texts: string[], embedOpts?: EmbedOptions): Promise<Vector[]> => {
        // Defensive empty-input guard: Voyage rejects empty `input` arrays;
        // skip the network call entirely.
        if (texts.length === 0) return [];
        const voyage = await voyageClientPromise!;
        const response = await voyage.embed(
          {
            input: texts,
            model: embedOpts?.model ?? voyageModel,
          },
          embedOpts?.signal ? { abortSignal: embedOpts.signal } : undefined,
        );
        // Voyage returns `data` items each with an optional `index` and
        // `embedding`. Sort by index when present so the returned vectors
        // align with the input texts even if the SDK reorders.
        const data = response.data ?? [];
        return data
          .slice()
          .sort((a, b) => (a.index ?? 0) - (b.index ?? 0))
          .map((d) => d.embedding ?? []);
      }
    : undefined;

  return {
    name: 'anthropic',

    async complete(prompt, opts) {
      const finalPrompt =
        opts?.format === 'json' ? `${prompt}${JSON_INSTRUCTION}` : prompt;
      const response = await client.messages.create({
        model,
        max_tokens: opts?.maxTokens ?? DEFAULT_MAX_TOKENS,
        ...(opts?.temperature !== undefined
          ? { temperature: opts.temperature }
          : {}),
        messages: [{ role: 'user', content: finalPrompt }],
      });
      // Concatenate text-typed content blocks; ignore tool_use / thinking /
      // anything else — complete() is the non-streaming path so we just want
      // the assistant's prose answer.
      return response.content
        .filter(
          (block): block is { type: 'text'; text: string } & typeof block =>
            block.type === 'text',
        )
        .map((block) => block.text)
        .join('');
    },

    stream(prompt, opts) {
      return anthropicStream(client, model, prompt, opts);
    },

    streamMessages(messages, opts) {
      return anthropicStreamMessages(client, model, messages, opts);
    },

    // `embed` is omitted (left as undefined on the literal) when no Voyage
    // config was supplied, so consumers see `provider.embed === undefined`.
    ...(embedFn ? { embed: embedFn } : {}),
  };
}

/**
 * Coalesce adjacent same-role messages by joining their content with a
 * double-newline. Anthropic's Messages API rejects two consecutive `user`
 * (or two consecutive `assistant`) entries — turns must strictly alternate.
 * This is a defensive merge so the engine can pass arbitrary message arrays
 * (including a corrective system follow-up that lands as a second `user`)
 * without bespoke knowledge of Anthropic's wire constraints.
 *
 * Operates only on `user` / `assistant` messages — `system` is extracted
 * out-of-band by the caller and is not present in `entries`.
 */
function coalesceAdjacent(
  entries: Array<{ role: 'user' | 'assistant'; content: string }>,
): Array<{ role: 'user' | 'assistant'; content: string }> {
  const out: Array<{ role: 'user' | 'assistant'; content: string }> = [];
  for (const entry of entries) {
    const last = out[out.length - 1];
    if (last && last.role === entry.role) {
      last.content = `${last.content}\n\n${entry.content}`;
    } else {
      out.push({ role: entry.role, content: entry.content });
    }
  }
  return out;
}

/**
 * Translate the contract's neutral tool shape into Anthropic's tool format.
 * Returned array is empty when no tools were supplied — the caller is
 * responsible for omitting the `tools` field entirely in that case (the API
 * rejects an empty array).
 */
function translateTools(
  tools: NonNullable<StreamOptions['tools']> | undefined,
): Array<{
  name: string;
  description?: string;
  input_schema: { type: 'object'; [k: string]: unknown };
}> {
  return (tools ?? []).map((t) => ({
    name: t.name,
    description: t.description,
    input_schema: t.parameters as { type: 'object'; [k: string]: unknown },
  }));
}

async function* anthropicStream(
  client: Anthropic,
  model: string,
  prompt: string,
  opts: StreamOptions = {},
): AsyncIterable<LLMStreamEvent> {
  const tools = translateTools(opts.tools);
  const finalPrompt =
    opts.format === 'json' ? `${prompt}${JSON_INSTRUCTION}` : prompt;

  const stream = (await client.messages.create(
    {
      model,
      max_tokens: opts.maxTokens ?? DEFAULT_MAX_TOKENS,
      ...(opts.temperature !== undefined ? { temperature: opts.temperature } : {}),
      messages: [{ role: 'user', content: finalPrompt }],
      ...(tools.length > 0 ? { tools } : {}),
      stream: true,
    },
    opts.signal ? { signal: opts.signal } : undefined,
  )) as unknown as AsyncIterable<RawMessageStreamEvent>;

  yield* consumeAnthropicStream(stream);
}

async function* anthropicStreamMessages(
  client: Anthropic,
  model: string,
  messages: LLMMessage[],
  opts: StreamOptions = {},
): AsyncIterable<LLMStreamEvent> {
  const tools = translateTools(opts.tools);

  // System messages live on Anthropic's top-level `system` field, NOT inside
  // the `messages` array. Concatenate all system entries with double-newline
  // so multi-system inputs (e.g., base instructions + per-call directives)
  // collapse cleanly.
  const systemText = messages
    .filter((m) => m.role === 'system')
    .map((m) => m.content)
    .join('\n\n');

  // The remaining user/assistant turns become the API's `messages`. Coalesce
  // adjacent same-role entries because the API rejects consecutive same-role
  // turns (e.g. two `user` in a row).
  const turns = coalesceAdjacent(
    messages
      .filter((m) => m.role !== 'system')
      .map((m) => ({ role: m.role as 'user' | 'assistant', content: m.content })),
  );

  // Apply JSON instruction by appending to the final user turn (the one the
  // model is about to respond to). If there is no user turn at the tail, fall
  // back to appending to the final turn regardless.
  let finalTurns = turns;
  if (opts.format === 'json' && finalTurns.length > 0) {
    const last = finalTurns[finalTurns.length - 1]!;
    finalTurns = [
      ...finalTurns.slice(0, -1),
      { role: last.role, content: `${last.content}${JSON_INSTRUCTION}` },
    ];
  }

  const stream = (await client.messages.create(
    {
      model,
      max_tokens: opts.maxTokens ?? DEFAULT_MAX_TOKENS,
      ...(opts.temperature !== undefined ? { temperature: opts.temperature } : {}),
      ...(systemText ? { system: systemText } : {}),
      messages: finalTurns,
      ...(tools.length > 0 ? { tools } : {}),
      stream: true,
    },
    opts.signal ? { signal: opts.signal } : undefined,
  )) as unknown as AsyncIterable<RawMessageStreamEvent>;

  yield* consumeAnthropicStream(stream);
}

/**
 * Consume an Anthropic SSE stream and yield neutral {@link LLMStreamEvent}s.
 *
 * Tool-call assembly: Anthropic streams tool_use blocks as a `content_block_start`
 * (carrying name + id) followed by N `content_block_delta` events of type
 * `input_json_delta` (carrying chunked JSON), terminated by `content_block_stop`.
 * We accumulate per-content-block-index buffers and emit a single tool_call
 * event at the matching content_block_stop.
 *
 * Always emits a final `{type: 'done'}` so consumers can release resources
 * deterministically — the contract requires it.
 */
async function* consumeAnthropicStream(
  stream: AsyncIterable<RawMessageStreamEvent>,
): AsyncIterable<LLMStreamEvent> {
  const toolBuffers = new Map<number, { name: string; argsParts: string[] }>();
  let finishReason: 'stop' | 'length' | 'aborted' | undefined;

  for await (const event of stream) {
    switch (event.type) {
      case 'content_block_start': {
        if (event.content_block.type === 'tool_use') {
          toolBuffers.set(event.index, {
            name: event.content_block.name,
            argsParts: [],
          });
        }
        break;
      }
      case 'content_block_delta': {
        if (event.delta.type === 'text_delta') {
          yield { type: 'text', delta: event.delta.text };
        } else if (event.delta.type === 'input_json_delta') {
          const buf = toolBuffers.get(event.index);
          if (buf) buf.argsParts.push(event.delta.partial_json);
        }
        break;
      }
      case 'content_block_stop': {
        const buf = toolBuffers.get(event.index);
        if (buf) {
          yield {
            type: 'tool_call',
            name: buf.name,
            arguments: buf.argsParts.join(''),
          };
          toolBuffers.delete(event.index);
        }
        break;
      }
      case 'message_delta': {
        finishReason = translateStopReason(event.delta.stop_reason);
        break;
      }
      // message_start, message_stop — nothing to emit; finalized below.
      default:
        break;
    }
  }

  yield { type: 'done', reason: finishReason ?? 'stop' };
}
