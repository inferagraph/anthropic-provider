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
