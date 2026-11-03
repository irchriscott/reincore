// Shared types and helpers for model selection. Split from ModelSelect.tsx
// so the component file only exports a component (react-refresh rule).

import type { CatalogModel, OllamaModel } from '../types'

export type ModelOption = {
  model_id: string
  model_name: string
}

// Convert a curated catalog to options.
export function catalogToOptions(catalog: CatalogModel[]): ModelOption[] {
  return catalog.map((m) => ({ model_id: m.model_id, model_name: m.model_name }))
}

// Convert an Ollama model list to options.
export function ollamaToOptions(models: OllamaModel[]): ModelOption[] {
  return models.map((m) => ({ model_id: m.id, model_name: m.name }))
}
