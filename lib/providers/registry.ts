import { createAnthropic } from "@ai-sdk/anthropic"
import { createDeepSeek } from "@ai-sdk/deepseek"
import { createGoogle } from "@ai-sdk/google"
import { createGroq } from "@ai-sdk/groq"
import { createMistral } from "@ai-sdk/mistral"
import { createOpenAI } from "@ai-sdk/openai"
import { createOpenAICompatible } from "@ai-sdk/openai-compatible"
import { createXai } from "@ai-sdk/xai"
import type { LanguageModel } from "ai"

import { getSetting, getSettingSource } from "@/lib/config"
import { agentContext } from "@/lib/ollama"
import { prepareOllamaRequest } from "@/lib/providers/ollama-request"
import {
  formatModelSpec,
  getProviderMeta,
  parseModelSpec,
  providers,
  type ProviderId,
} from "@/lib/providers/catalog"

export class ModelConfigError extends Error {
  name = "ModelConfigError"
}

export function getProviderCredentials(id: ProviderId) {
  return {
    apiKey: getSetting(`provider.${id}.apiKey`),
    baseUrl: getSetting(`provider.${id}.baseUrl`),
  }
}

// A provider counts as configured once it has what it needs to answer:
// a key for hosted ones, an endpoint for local and custom ones
export function isProviderConfigured(id: ProviderId) {
  const meta = getProviderMeta(id)
  if (!meta) return false
  const { apiKey, baseUrl } = getProviderCredentials(id)
  if (meta.requiresKey && !apiKey) return false
  if (meta.requiresBaseUrl && !baseUrl) return false
  // Local servers are assumed off until the person points Xiryn at them
  if (meta.local) {
    const source = getSettingSource(`provider.${id}.baseUrl`)
    return source === "local" || source === "env"
  }
  return true
}

export function configuredProviders() {
  return providers.filter((p) => isProviderConfigured(p.id))
}

// The model a turn runs on: the agent's own, else the workspace default,
// else the first suggestion of the first configured provider
export function resolveModelSpec(agentModel?: string | null) {
  if (agentModel) return agentModel
  const fallback = getSetting("model.default")
  if (fallback) return fallback
  const first = configuredProviders().find((p) => p.suggested.length > 0)
  return first ? formatModelSpec(first.id, first.suggested[0]) : null
}

export function createModel(spec: string): LanguageModel {
  const parsed = parseModelSpec(spec)
  if (!parsed) {
    throw new ModelConfigError(
      `"${spec}" is not a model Xiryn knows how to reach. Use provider/model, e.g. openai/gpt-6-luna.`
    )
  }

  const { provider, model, meta } = parsed
  const { apiKey, baseUrl } = getProviderCredentials(provider)

  if (meta.requiresKey && !apiKey) {
    throw new ModelConfigError(
      `${meta.name} has no API key. Add ${meta.keyEnv} to .env or paste a key in Settings → Providers.`
    )
  }
  if (meta.requiresBaseUrl && !baseUrl) {
    throw new ModelConfigError(
      `${meta.name} needs an endpoint. Set ${meta.baseUrlEnv} or add one in Settings → Providers.`
    )
  }

  // Only pass an endpoint when it differs from the SDK's own default
  const baseURL = baseUrl && baseUrl !== meta.defaultBaseUrl ? baseUrl : undefined

  switch (provider) {
    case "openai":
      return createOpenAI({ apiKey, baseURL })(model)
    case "anthropic":
      return createAnthropic({ apiKey, baseURL })(model)
    case "google":
      return createGoogle({ apiKey, baseURL })(model)
    case "xai":
      return createXai({ apiKey, baseURL })(model)
    case "groq":
      return createGroq({ apiKey, baseURL })(model)
    case "mistral":
      return createMistral({ apiKey, baseURL })(model)
    case "deepseek":
      return createDeepSeek({ apiKey, baseURL })(model)
    case "openrouter":
    case "ollama":
    case "lmstudio":
    case "custom":
      return createOpenAICompatible({
        name: provider,
        baseURL: baseUrl || meta.defaultBaseUrl,
        apiKey: apiKey || undefined,
        includeUsage: true,
        supportsMultiPartToolContent: provider === "ollama",
        ...(provider === "ollama" && {
          transformRequestBody: (args: Record<string, unknown>) =>
            prepareOllamaRequest(args, agentContext()),
        }),
      }).chatModel(model)
  }
}
