import { describe, it, expect, vi, beforeEach } from 'vitest';
import { AnthropicProvider } from '../src/AnthropicProvider.js';

vi.mock('@anthropic-ai/sdk', () => {
  return {
    default: vi.fn().mockImplementation(() => ({
      messages: {
        create: vi.fn().mockResolvedValue({
          content: [{ type: 'text', text: 'Test response about Adam and Eve.' }],
          usage: { input_tokens: 50, output_tokens: 10 },
        }),
      },
    })),
  };
});

describe('AnthropicProvider', () => {
  let provider: AnthropicProvider;

  beforeEach(() => {
    provider = new AnthropicProvider({ apiKey: 'test-key' });
  });

  it('should have name anthropic', () => {
    expect(provider.name).toBe('anthropic');
  });

  it('should be configured', () => {
    expect(provider.isConfigured()).toBe(true);
  });

  it('should complete a request', async () => {
    const result = await provider.complete({
      messages: [
        { role: 'system', content: 'You are a Bible scholar.' },
        { role: 'user', content: 'Who is Adam?' },
      ],
    });
    expect(result.content).toBe('Test response about Adam and Eve.');
    expect(result.usage?.inputTokens).toBe(50);
    expect(result.usage?.outputTokens).toBe(10);
  });

  it('should handle missing text block', async () => {
    const { default: Anthropic } = await import('@anthropic-ai/sdk');
    const mockInstance = new (Anthropic as any)();
    mockInstance.messages.create.mockResolvedValueOnce({
      content: [],
      usage: { input_tokens: 10, output_tokens: 0 },
    });

    const customProvider = new AnthropicProvider({ apiKey: 'test-key' });
    (customProvider as any).client = mockInstance;

    const result = await customProvider.complete({
      messages: [{ role: 'user', content: 'test' }],
    });
    expect(result.content).toBe('');
  });

  it('should accept custom model and maxTokens', () => {
    const custom = new AnthropicProvider({
      apiKey: 'test-key',
      model: 'claude-opus-4-20250514',
      maxTokens: 2048,
    });
    expect(custom.name).toBe('anthropic');
  });
});
