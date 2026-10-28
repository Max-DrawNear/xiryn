// Ollama can silently truncate oversized input. Count UTF-8 bytes as a
// conservative text-token bound, reserve output space, and remove only whole
// older user turns. Screenshots use a separate allowance instead of base64 bytes.
type Message = { role: string; content?: unknown; [key: string]: unknown }

export function requestCost(value: unknown) {
  let images = 0
  const serialized =
    JSON.stringify(value, (_key, part) => {
      if (part && typeof part === "object" && part.type === "image_url") {
        images++
        return { type: "image_url", image_url: "[image]" }
      }
      return part
    }) ?? ""
  return Buffer.byteLength(serialized, "utf8") + images * 4096
}

export function prepareOllamaRequest(args: Record<string, unknown>, context: number) {
  const messages = args.messages as Message[]
  if (!Array.isArray(messages)) return args
  const maxOutput = typeof args.max_tokens === "number" ? args.max_tokens : 4096
  const budget = context - maxOutput - 1024
  const fixed = messages.filter((m) => m.role === "system" || m.role === "developer")
  const conversation = messages.filter(
    (m) => m.role !== "system" && m.role !== "developer"
  )
  const groups: Message[][] = []
  for (const message of conversation) {
    if (message.role === "user" || groups.length === 0) groups.push([])
    groups[groups.length - 1].push(message)
  }
  const notice: Message = {
    role: "system",
    content:
      "Some older conversation turns were omitted to fit the local model context. Do not invent missing details. Use saved memory, memory_search, or project files; ask if the needed information is unavailable.",
  }
  let dropped = 0
  let kept = groups.flat()
  const cost = () =>
    requestCost({
      ...args,
      messages: [...fixed, ...(dropped ? [notice] : []), ...kept],
      max_tokens: maxOutput,
    })
  while (cost() > budget && groups.length > 1) {
    groups.shift()
    dropped++
    kept = groups.flat()
  }
  if (cost() > budget) {
    throw new Error(
      "The current task exceeds the local model context budget. Start a fresh thread or request a smaller file/page section. Saved memory is retained; the task was not silently truncated."
    )
  }
  const qwen = /^qwen3[.]5(?=[:/]|$)/i.test(String(args.model))
  return {
    ...args,
    messages: [...fixed, ...(dropped ? [notice] : []), ...kept],
    max_tokens: maxOutput,
    ...(qwen && { reasoning_effort: "none", temperature: 0.6 }),
  }
}
