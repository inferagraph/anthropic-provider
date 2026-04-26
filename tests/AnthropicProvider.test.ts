import { describe, it, expect, vi, beforeEach } from 'vitest';
import { AnthropicProvider } from '../src/AnthropicProvider.js';

const mockCreate = vi.fn();
const mockStream = vi.fn();

vi.mock('@anthropic-ai/sdk', () => {
  return {
    default: vi.fn().mockImplementation(() => ({
      messages: {
        create: (...args: unknown[]) => mockCreate(...args),
        stream: (...args: unknown[]) => mockStream(...args),
      },
    })),
  };
});

describe('AnthropicProvider', () => {
  let provider: AnthropicProvider;

  beforeEach(() => {
    vi.clearAllMocks();
    mockCreate.mockResolvedValue({
      content: [{ type: 'text', text: 'Test response about Adam and Eve.' }],
      usage: { input_tokens: 50, output_tokens: 10 },
    });
    provider = new AnthropicProvider({ apiKey: 'test-key' });
  });

  it('should have name anthropic', () => {
    expect(provider.name).toBe('anthropic');
  });

  it('should be configured', () => {
    expect(provider.isConfigured()).toBe(true);
  });

  it('should complete a request with system message', async () => {
    const result = await provider.complete({
      messages: [
        { role: 'system', content: 'You are a Bible scholar.' },
        { role: 'user', content: 'Who is Adam?' },
      ],
    });
    expect(result.content).toBe('Test response about Adam and Eve.');
    expect(result.usage?.inputTokens).toBe(50);
    expect(result.usage?.outputTokens).toBe(10);

    expect(mockCreate).toHaveBeenCalledWith(
      expect.objectContaining({ system: 'You are a Bible scholar.' }),
    );
  });

  it('should complete a request without system message', async () => {
    await provider.complete({
      messages: [{ role: 'user', content: 'Who is Adam?' }],
    });

    const callArgs = mockCreate.mock.calls[0][0];
    expect(callArgs).not.toHaveProperty('system');
  });

  it('should pass maxTokens from request when provided', async () => {
    await provider.complete({
      messages: [{ role: 'user', content: 'test' }],
      maxTokens: 512,
    });
    expect(mockCreate).toHaveBeenCalledWith(
      expect.objectContaining({ max_tokens: 512 }),
    );
  });

  it('should handle missing text block', async () => {
    mockCreate.mockResolvedValueOnce({
      content: [],
      usage: { input_tokens: 10, output_tokens: 0 },
    });

    const result = await provider.complete({
      messages: [{ role: 'user', content: 'test' }],
    });
    expect(result.content).toBe('');
  });

  it('should handle non-text content blocks', async () => {
    mockCreate.mockResolvedValueOnce({
      content: [{ type: 'tool_use', id: 'tool_1' }],
      usage: { input_tokens: 10, output_tokens: 5 },
    });

    const result = await provider.complete({
      messages: [{ role: 'user', content: 'test' }],
    });
    expect(result.content).toBe('');
  });

  it('should accept custom model, maxTokens, and baseURL', () => {
    const custom = new AnthropicProvider({
      apiKey: 'test-key',
      model: 'claude-opus-4-20250514',
      maxTokens: 2048,
      baseURL: 'https://custom.anthropic.com',
    });
    expect(custom.name).toBe('anthropic');
  });

  describe('stream', () => {
    it('should yield text chunks and done for text_delta events', async () => {
      const asyncEvents = (async function* () {
        yield { type: 'content_block_delta', delta: { type: 'text_delta', text: 'Hello' } };
        yield { type: 'content_block_delta', delta: { type: 'text_delta', text: ' world' } };
      })();

      mockStream.mockReturnValueOnce(asyncEvents);

      const chunks = [];
      for await (const chunk of provider.stream({
        messages: [{ role: 'user', content: 'Hi' }],
      })) {
        chunks.push(chunk);
      }

      expect(chunks).toEqual([
        { type: 'text', content: 'Hello' },
        { type: 'text', content: ' world' },
        { type: 'done', content: '' },
      ]);
    });

    it('should skip non-text-delta events', async () => {
      const asyncEvents = (async function* () {
        yield { type: 'message_start', message: {} };
        yield { type: 'content_block_delta', delta: { type: 'text_delta', text: 'data' } };
        yield { type: 'content_block_delta', delta: { type: 'input_json_delta', partial_json: '{}' } };
        yield { type: 'message_stop' };
      })();

      mockStream.mockReturnValueOnce(asyncEvents);

      const chunks = [];
      for await (const chunk of provider.stream({
        messages: [{ role: 'user', content: 'Hi' }],
      })) {
        chunks.push(chunk);
      }

      expect(chunks).toEqual([
        { type: 'text', content: 'data' },
        { type: 'done', content: '' },
      ]);
    });

    it('should yield error chunk on failure', async () => {
      mockStream.mockImplementationOnce(() => {
        throw new Error('API rate limit');
      });

      const chunks = [];
      for await (const chunk of provider.stream({
        messages: [{ role: 'user', content: 'Hi' }],
      })) {
        chunks.push(chunk);
      }

      expect(chunks).toEqual([
        { type: 'error', content: 'API rate limit' },
      ]);
    });

    it('should yield error chunk with stringified non-Error', async () => {
      mockStream.mockImplementationOnce(() => {
        throw 'something went wrong';
      });

      const chunks = [];
      for await (const chunk of provider.stream({
        messages: [{ role: 'user', content: 'Hi' }],
      })) {
        chunks.push(chunk);
      }

      expect(chunks).toEqual([
        { type: 'error', content: 'something went wrong' },
      ]);
    });

    it('should pass system message and maxTokens in stream', async () => {
      const asyncEvents = (async function* () {
        yield { type: 'content_block_delta', delta: { type: 'text_delta', text: 'ok' } };
      })();

      mockStream.mockReturnValueOnce(asyncEvents);

      for await (const _chunk of provider.stream({
        messages: [
          { role: 'system', content: 'Be helpful' },
          { role: 'user', content: 'Hi' },
        ],
        maxTokens: 256,
      })) {
        // consume
      }

      expect(mockStream).toHaveBeenCalledWith(
        expect.objectContaining({
          system: 'Be helpful',
          max_tokens: 256,
        }),
      );
    });

    it('should not include system key when no system message exists', async () => {
      const asyncEvents = (async function* () {
        yield { type: 'content_block_delta', delta: { type: 'text_delta', text: 'ok' } };
      })();

      mockStream.mockReturnValueOnce(asyncEvents);

      for await (const _chunk of provider.stream({
        messages: [{ role: 'user', content: 'Hi' }],
      })) {
        // consume
      }

      const callArgs = mockStream.mock.calls[0][0];
      expect(callArgs).not.toHaveProperty('system');
    });
  });
});
