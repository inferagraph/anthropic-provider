import Anthropic from '@anthropic-ai/sdk';
import { LLMProvider } from '@inferagraph/core';
import type { LLMCompletionRequest, LLMCompletionResponse, LLMStreamChunk } from '@inferagraph/core';
import type { AnthropicProviderConfig } from './types.js';

const DEFAULT_MODEL = 'claude-sonnet-4-20250514';
const DEFAULT_MAX_TOKENS = 1024;

export class AnthropicProvider extends LLMProvider {
  readonly name = 'anthropic';
  private readonly client: Anthropic;
  private readonly model: string;
  private readonly maxTokens: number;

  constructor(config: AnthropicProviderConfig) {
    super();
    this.client = new Anthropic({
      apiKey: config.apiKey,
      ...(config.baseURL ? { baseURL: config.baseURL } : {}),
    });
    this.model = config.model ?? DEFAULT_MODEL;
    this.maxTokens = config.maxTokens ?? DEFAULT_MAX_TOKENS;
  }

  async complete(request: LLMCompletionRequest): Promise<LLMCompletionResponse> {
    const systemMessage = request.messages.find((m) => m.role === 'system');
    const userMessages = request.messages
      .filter((m) => m.role !== 'system')
      .map((m) => ({ role: m.role as 'user' | 'assistant', content: m.content }));

    const response = await this.client.messages.create({
      model: this.model,
      max_tokens: request.maxTokens ?? this.maxTokens,
      ...(systemMessage ? { system: systemMessage.content } : {}),
      messages: userMessages,
    });

    const textBlock = response.content.find((block) => block.type === 'text');
    return {
      content: textBlock?.text ?? '',
      usage: {
        inputTokens: response.usage.input_tokens,
        outputTokens: response.usage.output_tokens,
      },
    };
  }

  async *stream(request: LLMCompletionRequest): AsyncIterable<LLMStreamChunk> {
    const systemMessage = request.messages.find(m => m.role === 'system');
    const nonSystemMessages = request.messages.filter(m => m.role !== 'system');

    try {
      const stream = this.client.messages.stream({
        model: this.model,
        max_tokens: request.maxTokens ?? this.maxTokens,
        ...(systemMessage && { system: systemMessage.content }),
        messages: nonSystemMessages.map(m => ({
          role: m.role as 'user' | 'assistant',
          content: m.content,
        })),
      });

      for await (const event of stream) {
        if (event.type === 'content_block_delta' && event.delta.type === 'text_delta') {
          yield { type: 'text' as const, content: event.delta.text };
        }
      }
      yield { type: 'done' as const, content: '' };
    } catch (error) {
      yield { type: 'error' as const, content: error instanceof Error ? error.message : String(error) };
    }
  }

  isConfigured(): boolean {
    return true;
  }
}
