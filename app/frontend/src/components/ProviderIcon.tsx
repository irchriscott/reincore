// ProviderIcon: brand-aware icon for each supported LLM provider.
// Falls back to the generic Cpu icon when no brand icon is available.
// Mirrors reincore-frontend/src/components/ProviderIcon.tsx.

import {
  OpenAI,
  Anthropic,
  GithubCopilot,
  Gemini,
  Groq,
  XAI,
  DeepSeek,
  Ollama,
  Azure,
} from '@lobehub/icons'
import { Cpu, type LucideProps } from 'lucide-react'

// Every supported provider now has a dedicated brand icon via @lobehub/icons.
const ICONS: Partial<Record<string, React.ComponentType<React.SVGProps<SVGSVGElement>>>> = {
  openai: OpenAI,
  anthropic: Anthropic,
  gemini: Gemini,
  azure: Azure,
  groq: Groq,
  xai: XAI,
  deepseek: DeepSeek,
  ollama: Ollama,
  copilot: GithubCopilot,
}

interface ProviderIconProps extends Omit<LucideProps, 'ref'> {
  id: string
  fallback?: React.ComponentType<LucideProps>
}

export function ProviderIcon({
  id,
  fallback: Fallback = Cpu,
  ...props
}: ProviderIconProps) {
  const Icon = ICONS[id]
  if (!Icon) {
    return <Fallback {...props} />
  }
  return <Icon {...props} />
}

// Re-export the icon mapping so callers can build color-aware variants or
// check availability without duplicating the table.
export const PROVIDER_ICON_COMPONENTS = ICONS