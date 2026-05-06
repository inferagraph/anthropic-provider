# @inferagraph/anthropic-provider

Anthropic Claude provider plugin for [@inferagraph/core](https://github.com/inferagraph/core), with optional [Voyage AI](https://www.voyageai.com/) embeddings.

## Installation

```bash
pnpm add @inferagraph/anthropic-provider @inferagraph/core
```

## Chat-only usage

```ts
import { anthropicProvider } from '@inferagraph/anthropic-provider';
import { InferaGraph } from '@inferagraph/core/react';

<InferaGraph
  data={data}
  llm={anthropicProvider({
    apiKey: process.env.ANTHROPIC_API_KEY!,
    model: 'claude-sonnet-4-20250514',
  })}
/>
```

`complete()` and `stream()` are wired to the Anthropic Messages API. Tool calls stream as `tool_call` events; text deltas stream as `text` events. Streams always end with `{ type: 'done' }`.

## `streamMessages(messages, opts)` (recommended)

`stream(prompt: string)` accepts a single user prompt. `streamMessages(messages)` accepts a structured conversation array, which unlocks:

- **`system` role** for system prompts. Tool-use-trained Claude models heavily discount instructions delivered as user-role content; passing them under `system` keeps directives where the model is trained to obey them. (Better than prepending to the user message.)
- **`assistant` role** to replay prior model turns — multi-turn conversation memory, corrective-retry flows after malformed tool calls, etc.
- **Multi-turn conversations** as a sequence of alternating `user` / `assistant` turns following an optional leading `system` turn.

Signature (peer dep `@inferagraph/core@^0.8.0` exports the `LLMMessage` / `LLMRole` types):

```ts
import type { LLMMessage, LLMRole } from '@inferagraph/core';

provider.streamMessages(
  messages: LLMMessage[],
  opts?: StreamOptions,
): AsyncIterable<LLMStreamEvent>;
```

Example — system prompt plus a 2-turn exchange:

```ts
import { anthropicProvider } from '@inferagraph/anthropic-provider';
import type { LLMMessage } from '@inferagraph/core';

const provider = anthropicProvider({
  apiKey: process.env.ANTHROPIC_API_KEY!,
  model: 'claude-sonnet-4-20250514',
});

const messages: LLMMessage[] = [
  { role: 'system', content: 'You are a concise assistant. Reply in one sentence.' },
  { role: 'user', content: 'Who wrote the Iliad?' },
  { role: 'assistant', content: 'Tradition attributes the Iliad to Homer.' },
  { role: 'user', content: 'And the Odyssey?' },
];

for await (const ev of provider.streamMessages!(messages)) {
  if (ev.type === 'text') process.stdout.write(ev.delta);
  if (ev.type === 'done') break;
}
```

The Anthropic SDK lifts `system` into a top-level field on the Messages API call rather than keeping it inline; the provider handles that transparently. Pass `system` as a normal entry in the `messages` array — it is routed to the SDK's `system` parameter, while `user` / `assistant` turns flow into the SDK's `messages` array. Output is identical to other providers.

### Back-compat

`stream(prompt)` still works and is unchanged. It is internally a thin wrapper that calls `streamMessages([{ role: 'user', content: prompt }])`, so single-prompt behavior is identical. New consumers should prefer `streamMessages` whenever a system prompt or prior turns are involved.

## Embeddings via Voyage AI

Anthropic does **not** expose a native embeddings endpoint. [Voyage AI](https://platform.claude.com/docs/en/build-with-claude/embeddings) is Anthropic's officially recommended embedding partner. Pass an optional `voyage` config to enable embedding support:

```ts
anthropicProvider({
  apiKey: process.env.ANTHROPIC_API_KEY!,
  voyage: {
    apiKey: process.env.VOYAGE_API_KEY!,
    model: 'voyage-3.5', // optional; default 'voyage-3.5'
  },
});
```

When `voyage` is omitted, the returned `LLMProvider` has `embed === undefined`. Chat still works; embedding-dependent features (semantic search, similarity highlight) are simply unavailable.

### Recommended Voyage models

| Model | When to use |
|-------|-------------|
| `voyage-3.5` | General-purpose default. 1024-dim, fast, low cost. |
| `voyage-3-large` | Higher quality at ~2× the cost. |
| `voyage-code-3` | Tuned for source code retrieval. |

Get a Voyage API key at [voyageai.com](https://www.voyageai.com/).

### Per-call model overrides

```ts
await provider.embed!(texts, { model: 'voyage-code-3' });
```

## Mix-and-match providers

You can keep Anthropic for chat and use a different provider for embeddings (e.g. `@inferagraph/openai-provider`'s OpenAI embeddings). The `LLMProvider` contract is structural; consumers may compose any combination they like.

## License

MIT
