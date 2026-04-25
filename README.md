# @inferagraph/anthropic-provider

Anthropic Claude provider plugin for [@inferagraph/core](https://github.com/inferagraph/core).

## Installation

```bash
pnpm add @inferagraph/anthropic-provider @inferagraph/core
```

## Usage

```typescript
import { AnthropicProvider } from '@inferagraph/anthropic-provider';

const provider = new AnthropicProvider({
  apiKey: 'your-api-key',
  model: 'claude-sonnet-4-20250514',
});
```

## License

MIT
