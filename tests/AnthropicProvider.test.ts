import { describe, it, expect, vi, beforeEach } from 'vitest';
import type Anthropic from '@anthropic-ai/sdk';
import type { VoyageAIClient } from 'voyageai';
import type { LLMStreamEvent } from '@inferagraph/core';
import { anthropicProvider } from '../src/index.js';

/**
 * Build a minimal mock Anthropic client. The provider only ever touches
 * `client.messages.create`, so we expose just that surface. `create` is a
 * vi.fn() that the tests configure per-case (resolve to a non-streaming
 * Message, or to an async iterable of stream events).
 */
function buildMockAnthropic(): {
  client: Anthropic;
  create: ReturnType<typeof vi.fn>;
} {
  const create = vi.fn();
  const client = {
    messages: {
      create,
    },
  } as unknown as Anthropic;
  return { client, create };
}

/**
 * Build a minimal mock Voyage client. The provider only touches
 * `client.embed`, which the SDK types as returning an `HttpResponsePromise`
 * (a Promise subclass). We satisfy it with a regular vi.fn() that returns a
 * resolved promise — `await` works the same.
 */
function buildMockVoyage(): {
  client: VoyageAIClient;
  embed: ReturnType<typeof vi.fn>;
} {
  const embed = vi.fn();
  const client = {
    embed,
  } as unknown as VoyageAIClient;
  return { client, embed };
}

/** Helper: build an async iterable from a fixed array of stream events. */
function asyncIterableOf<T>(items: T[]): AsyncIterable<T> {
  return (async function* () {
    for (const it of items) yield it;
  })();
}

/** Helper: drain an LLMStreamEvent async iterable into an array. */
async function collect(
  stream: AsyncIterable<LLMStreamEvent>,
): Promise<LLMStreamEvent[]> {
  const out: LLMStreamEvent[] = [];
  for await (const ev of stream) out.push(ev);
  return out;
}

describe('anthropicProvider', () => {
  let client: Anthropic;
  let create: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    ({ client, create } = buildMockAnthropic());
  });

  describe('factory', () => {
    it('exposes name "anthropic"', () => {
      const provider = anthropicProvider({ apiKey: 'k', client });
      expect(provider.name).toBe('anthropic');
    });

    it('uses the provided client when given (apiKey ignored)', async () => {
      create.mockResolvedValueOnce({
        content: [{ type: 'text', text: 'hi', citations: null }],
      });
      const provider = anthropicProvider({ apiKey: 'unused', client });
      const result = await provider.complete('ping');
      expect(result).toBe('hi');
      expect(create).toHaveBeenCalledOnce();
    });

    it('embed is undefined when voyage config is omitted', () => {
      const provider = anthropicProvider({ apiKey: 'k', client });
      expect(provider.embed).toBeUndefined();
    });

    it('embed is defined when voyage config is provided', () => {
      const { client: voyage } = buildMockVoyage();
      const provider = anthropicProvider({
        apiKey: 'k',
        client,
        voyage: { apiKey: 'v', client: voyage },
      });
      expect(typeof provider.embed).toBe('function');
    });
  });

  describe('complete()', () => {
    it('sends the prompt as a user message and returns concatenated text', async () => {
      create.mockResolvedValueOnce({
        content: [
          { type: 'text', text: 'hello ', citations: null },
          { type: 'text', text: 'world', citations: null },
        ],
      });
      const provider = anthropicProvider({ apiKey: 'k', client });
      const result = await provider.complete('Who is Adam?');
      expect(result).toBe('hello world');
      expect(create).toHaveBeenCalledWith({
        model: 'claude-sonnet-4-20250514',
        max_tokens: 4096,
        messages: [{ role: 'user', content: 'Who is Adam?' }],
      });
    });

    it('uses a custom model when configured', async () => {
      create.mockResolvedValueOnce({
        content: [{ type: 'text', text: 'ok', citations: null }],
      });
      const provider = anthropicProvider({
        apiKey: 'k',
        model: 'claude-opus-4-20250514',
        client,
      });
      await provider.complete('x');
      expect(create).toHaveBeenCalledWith(
        expect.objectContaining({ model: 'claude-opus-4-20250514' }),
      );
    });

    it('passes maxTokens and temperature when provided', async () => {
      create.mockResolvedValueOnce({
        content: [{ type: 'text', text: '', citations: null }],
      });
      const provider = anthropicProvider({ apiKey: 'k', client });
      await provider.complete('x', { maxTokens: 256, temperature: 0.3 });
      expect(create).toHaveBeenCalledWith(
        expect.objectContaining({ max_tokens: 256, temperature: 0.3 }),
      );
    });

    it('falls back to default max_tokens when not provided', async () => {
      create.mockResolvedValueOnce({
        content: [{ type: 'text', text: '', citations: null }],
      });
      const provider = anthropicProvider({ apiKey: 'k', client });
      await provider.complete('x');
      const args = create.mock.calls[0]![0] as { max_tokens: number };
      expect(args.max_tokens).toBe(4096);
    });

    it("appends JSON-only instruction when format='json'", async () => {
      create.mockResolvedValueOnce({
        content: [{ type: 'text', text: '{}', citations: null }],
      });
      const provider = anthropicProvider({ apiKey: 'k', client });
      await provider.complete('Give me data', { format: 'json' });
      const args = create.mock.calls[0]![0] as {
        messages: Array<{ role: string; content: string }>;
      };
      expect(args.messages[0]!.content).toContain('Give me data');
      expect(args.messages[0]!.content).toContain('valid JSON only');
    });

    it("does NOT append JSON instruction when format='text'", async () => {
      create.mockResolvedValueOnce({
        content: [{ type: 'text', text: 'plain', citations: null }],
      });
      const provider = anthropicProvider({ apiKey: 'k', client });
      await provider.complete('plain prompt', { format: 'text' });
      const args = create.mock.calls[0]![0] as {
        messages: Array<{ role: string; content: string }>;
      };
      expect(args.messages[0]!.content).toBe('plain prompt');
    });

    it('returns empty string when content is empty', async () => {
      create.mockResolvedValueOnce({ content: [] });
      const provider = anthropicProvider({ apiKey: 'k', client });
      const result = await provider.complete('x');
      expect(result).toBe('');
    });

    it('ignores non-text content blocks (tool_use, thinking)', async () => {
      create.mockResolvedValueOnce({
        content: [
          { type: 'tool_use', id: 't1', name: 'f', input: {} },
          { type: 'text', text: 'visible', citations: null },
          { type: 'thinking', thinking: 'hidden', signature: 'sig' },
        ],
      });
      const provider = anthropicProvider({ apiKey: 'k', client });
      const result = await provider.complete('x');
      expect(result).toBe('visible');
    });
  });

  describe('stream()', () => {
    it('yields text events for text_delta, then a done event', async () => {
      create.mockResolvedValueOnce(
        asyncIterableOf([
          {
            type: 'content_block_delta',
            index: 0,
            delta: { type: 'text_delta', text: 'Hello' },
          },
          {
            type: 'content_block_delta',
            index: 0,
            delta: { type: 'text_delta', text: ' world' },
          },
          {
            type: 'message_delta',
            delta: { stop_reason: 'end_turn', stop_sequence: null },
            usage: { output_tokens: 2 },
          },
          { type: 'message_stop' },
        ]),
      );
      const provider = anthropicProvider({ apiKey: 'k', client });
      const events = await collect(provider.stream('hi'));
      expect(events).toEqual([
        { type: 'text', delta: 'Hello' },
        { type: 'text', delta: ' world' },
        { type: 'done', reason: 'stop' },
      ]);
    });

    it('emits done with reason "length" when stop_reason is max_tokens', async () => {
      create.mockResolvedValueOnce(
        asyncIterableOf([
          {
            type: 'content_block_delta',
            index: 0,
            delta: { type: 'text_delta', text: 'partial' },
          },
          {
            type: 'message_delta',
            delta: { stop_reason: 'max_tokens', stop_sequence: null },
            usage: { output_tokens: 5 },
          },
          { type: 'message_stop' },
        ]),
      );
      const provider = anthropicProvider({ apiKey: 'k', client });
      const events = await collect(provider.stream('hi'));
      expect(events.at(-1)).toEqual({ type: 'done', reason: 'length' });
    });

    it('falls back to "stop" when stop_reason was never sent', async () => {
      create.mockResolvedValueOnce(
        asyncIterableOf([
          {
            type: 'content_block_delta',
            index: 0,
            delta: { type: 'text_delta', text: 'a' },
          },
        ]),
      );
      const provider = anthropicProvider({ apiKey: 'k', client });
      const events = await collect(provider.stream('hi'));
      expect(events.at(-1)).toEqual({ type: 'done', reason: 'stop' });
    });

    it('accumulates a single tool_call across multiple input_json_delta events and emits at content_block_stop', async () => {
      create.mockResolvedValueOnce(
        asyncIterableOf([
          {
            type: 'content_block_start',
            index: 0,
            content_block: {
              type: 'tool_use',
              id: 'toolu_1',
              name: 'apply_filter',
              input: {},
            },
          },
          {
            type: 'content_block_delta',
            index: 0,
            delta: { type: 'input_json_delta', partial_json: '{"era"' },
          },
          {
            type: 'content_block_delta',
            index: 0,
            delta: { type: 'input_json_delta', partial_json: ':"patriarchs"}' },
          },
          { type: 'content_block_stop', index: 0 },
          {
            type: 'message_delta',
            delta: { stop_reason: 'tool_use', stop_sequence: null },
            usage: { output_tokens: 5 },
          },
          { type: 'message_stop' },
        ]),
      );
      const provider = anthropicProvider({ apiKey: 'k', client });
      const events = await collect(provider.stream('show patriarchs'));
      expect(events).toEqual([
        {
          type: 'tool_call',
          name: 'apply_filter',
          arguments: '{"era":"patriarchs"}',
        },
        { type: 'done', reason: 'stop' },
      ]);
    });

    it('emits text deltas before tool_calls in the order they arrive', async () => {
      create.mockResolvedValueOnce(
        asyncIterableOf([
          {
            type: 'content_block_start',
            index: 0,
            content_block: { type: 'text', text: '', citations: null },
          },
          {
            type: 'content_block_delta',
            index: 0,
            delta: { type: 'text_delta', text: 'thinking...' },
          },
          { type: 'content_block_stop', index: 0 },
          {
            type: 'content_block_start',
            index: 1,
            content_block: {
              type: 'tool_use',
              id: 'toolu_2',
              name: 'highlight',
              input: {},
            },
          },
          {
            type: 'content_block_delta',
            index: 1,
            delta: { type: 'input_json_delta', partial_json: '{"id":"a"}' },
          },
          { type: 'content_block_stop', index: 1 },
          {
            type: 'message_delta',
            delta: { stop_reason: 'tool_use', stop_sequence: null },
            usage: { output_tokens: 3 },
          },
          { type: 'message_stop' },
        ]),
      );
      const provider = anthropicProvider({ apiKey: 'k', client });
      const events = await collect(provider.stream('do it'));
      expect(events).toEqual([
        { type: 'text', delta: 'thinking...' },
        { type: 'tool_call', name: 'highlight', arguments: '{"id":"a"}' },
        { type: 'done', reason: 'stop' },
      ]);
    });

    it('handles parallel tool_use blocks keyed by index', async () => {
      create.mockResolvedValueOnce(
        asyncIterableOf([
          {
            type: 'content_block_start',
            index: 0,
            content_block: {
              type: 'tool_use',
              id: 't1',
              name: 'highlight',
              input: {},
            },
          },
          {
            type: 'content_block_start',
            index: 1,
            content_block: {
              type: 'tool_use',
              id: 't2',
              name: 'focus',
              input: {},
            },
          },
          {
            type: 'content_block_delta',
            index: 0,
            delta: { type: 'input_json_delta', partial_json: '{"id":"a"}' },
          },
          {
            type: 'content_block_delta',
            index: 1,
            delta: { type: 'input_json_delta', partial_json: '{"id":"b"}' },
          },
          { type: 'content_block_stop', index: 0 },
          { type: 'content_block_stop', index: 1 },
          {
            type: 'message_delta',
            delta: { stop_reason: 'tool_use', stop_sequence: null },
            usage: { output_tokens: 4 },
          },
        ]),
      );
      const provider = anthropicProvider({ apiKey: 'k', client });
      const events = await collect(provider.stream('do both'));
      const toolCalls = events.filter((e) => e.type === 'tool_call');
      expect(toolCalls).toEqual([
        { type: 'tool_call', name: 'highlight', arguments: '{"id":"a"}' },
        { type: 'tool_call', name: 'focus', arguments: '{"id":"b"}' },
      ]);
    });

    it('translates StreamOptions.tools into Anthropic input_schema format', async () => {
      create.mockResolvedValueOnce(asyncIterableOf([]));
      const provider = anthropicProvider({ apiKey: 'k', client });
      await collect(
        provider.stream('hi', {
          tools: [
            {
              name: 'apply_filter',
              description: 'Restrict the visible set',
              parameters: {
                type: 'object',
                properties: { predicate: { type: 'string' } },
                required: ['predicate'],
              },
            },
          ],
        }),
      );
      expect(create).toHaveBeenCalledWith(
        expect.objectContaining({
          tools: [
            {
              name: 'apply_filter',
              description: 'Restrict the visible set',
              input_schema: {
                type: 'object',
                properties: { predicate: { type: 'string' } },
                required: ['predicate'],
              },
            },
          ],
        }),
        undefined,
      );
    });

    it('omits tools entirely when none are supplied', async () => {
      create.mockResolvedValueOnce(asyncIterableOf([]));
      const provider = anthropicProvider({ apiKey: 'k', client });
      await collect(provider.stream('hi'));
      const args = create.mock.calls[0]![0] as { tools?: unknown };
      expect(args.tools).toBeUndefined();
    });

    it('forwards the AbortSignal as a request option', async () => {
      const ctrl = new AbortController();
      create.mockResolvedValueOnce(asyncIterableOf([]));
      const provider = anthropicProvider({ apiKey: 'k', client });
      await collect(provider.stream('hi', { signal: ctrl.signal }));
      expect(create).toHaveBeenCalledWith(
        expect.objectContaining({ stream: true }),
        { signal: ctrl.signal },
      );
    });

    it('passes maxTokens and temperature into the streaming request', async () => {
      create.mockResolvedValueOnce(asyncIterableOf([]));
      const provider = anthropicProvider({ apiKey: 'k', client });
      await collect(
        provider.stream('hi', { maxTokens: 128, temperature: 0.7 }),
      );
      expect(create).toHaveBeenCalledWith(
        expect.objectContaining({
          max_tokens: 128,
          temperature: 0.7,
          stream: true,
        }),
        undefined,
      );
    });

    it("appends JSON instruction to the prompt when format='json'", async () => {
      create.mockResolvedValueOnce(asyncIterableOf([]));
      const provider = anthropicProvider({ apiKey: 'k', client });
      await collect(provider.stream('Give me data', { format: 'json' }));
      const args = create.mock.calls[0]![0] as {
        messages: Array<{ role: string; content: string }>;
      };
      expect(args.messages[0]!.content).toContain('Give me data');
      expect(args.messages[0]!.content).toContain('valid JSON only');
    });

    it('handles a stream that yields nothing at all and still emits done', async () => {
      create.mockResolvedValueOnce(asyncIterableOf([]));
      const provider = anthropicProvider({ apiKey: 'k', client });
      const events = await collect(provider.stream('hi'));
      expect(events).toEqual([{ type: 'done', reason: 'stop' }]);
    });

    it('drops content_block_stop for blocks that were never tool_use', async () => {
      create.mockResolvedValueOnce(
        asyncIterableOf([
          {
            type: 'content_block_start',
            index: 0,
            content_block: { type: 'text', text: '', citations: null },
          },
          {
            type: 'content_block_delta',
            index: 0,
            delta: { type: 'text_delta', text: 'hi' },
          },
          { type: 'content_block_stop', index: 0 },
          {
            type: 'message_delta',
            delta: { stop_reason: 'end_turn', stop_sequence: null },
            usage: { output_tokens: 1 },
          },
        ]),
      );
      const provider = anthropicProvider({ apiKey: 'k', client });
      const events = await collect(provider.stream('hi'));
      expect(events.find((e) => e.type === 'tool_call')).toBeUndefined();
    });
  });

  describe('embed() (Voyage integration)', () => {
    let voyageEmbed: ReturnType<typeof vi.fn>;
    let voyageClient: VoyageAIClient;

    beforeEach(() => {
      ({ client: voyageClient, embed: voyageEmbed } = buildMockVoyage());
    });

    function makeEmbedProvider(model?: string) {
      return anthropicProvider({
        apiKey: 'k',
        client,
        voyage: { apiKey: 'v', client: voyageClient, ...(model ? { model } : {}) },
      });
    }

    it('calls Voyage with the default voyage-3.5 model when no override is supplied', async () => {
      voyageEmbed.mockResolvedValueOnce({
        data: [{ index: 0, embedding: [0.1, 0.2] }],
      });
      const provider = makeEmbedProvider();
      const result = await provider.embed!(['hello']);
      expect(result).toEqual([[0.1, 0.2]]);
      expect(voyageEmbed).toHaveBeenCalledWith(
        { input: ['hello'], model: 'voyage-3.5' },
        undefined,
      );
    });

    it('honors config.voyage.model as the default', async () => {
      voyageEmbed.mockResolvedValueOnce({
        data: [{ index: 0, embedding: [0.5] }],
      });
      const provider = makeEmbedProvider('voyage-3-large');
      await provider.embed!(['x']);
      expect(voyageEmbed).toHaveBeenCalledWith(
        expect.objectContaining({ model: 'voyage-3-large' }),
        undefined,
      );
    });

    it('honors per-call EmbedOptions.model override over config default', async () => {
      voyageEmbed.mockResolvedValueOnce({
        data: [{ index: 0, embedding: [0.5] }],
      });
      const provider = makeEmbedProvider('voyage-3-large');
      await provider.embed!(['x'], { model: 'voyage-code-3' });
      expect(voyageEmbed).toHaveBeenCalledWith(
        expect.objectContaining({ model: 'voyage-code-3' }),
        undefined,
      );
    });

    it('returns embeddings sorted by index for batch input', async () => {
      voyageEmbed.mockResolvedValueOnce({
        data: [
          { index: 1, embedding: [0.2, 0.2] },
          { index: 0, embedding: [0.1, 0.1] },
          { index: 2, embedding: [0.3, 0.3] },
        ],
      });
      const provider = makeEmbedProvider();
      const result = await provider.embed!(['a', 'b', 'c']);
      expect(result).toEqual([
        [0.1, 0.1],
        [0.2, 0.2],
        [0.3, 0.3],
      ]);
    });

    it('returns [] without calling Voyage when input is empty', async () => {
      const provider = makeEmbedProvider();
      const result = await provider.embed!([]);
      expect(result).toEqual([]);
      expect(voyageEmbed).not.toHaveBeenCalled();
    });

    it('forwards AbortSignal as Voyage abortSignal request option', async () => {
      const ctrl = new AbortController();
      voyageEmbed.mockResolvedValueOnce({
        data: [{ index: 0, embedding: [0.1] }],
      });
      const provider = makeEmbedProvider();
      await provider.embed!(['x'], { signal: ctrl.signal });
      expect(voyageEmbed).toHaveBeenCalledWith(
        expect.objectContaining({ input: ['x'] }),
        { abortSignal: ctrl.signal },
      );
    });

    it('handles missing embedding fields gracefully (returns [])', async () => {
      voyageEmbed.mockResolvedValueOnce({
        data: [{ index: 0 }],
      });
      const provider = makeEmbedProvider();
      const result = await provider.embed!(['x']);
      expect(result).toEqual([[]]);
    });

    it('handles a response with no data array', async () => {
      voyageEmbed.mockResolvedValueOnce({});
      const provider = makeEmbedProvider();
      const result = await provider.embed!(['x']);
      expect(result).toEqual([]);
    });
  });
});
