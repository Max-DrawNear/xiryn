import fs from "node:fs"
import assert from "node:assert/strict"
import ts from "typescript"
const source = fs.readFileSync(
  new URL("../lib/providers/ollama-request.ts", import.meta.url),
  "utf8"
)
const output = ts.transpile(source, {
  module: ts.ModuleKind.CommonJS,
  target: ts.ScriptTarget.ES2022,
})
const api = {}
new Function("exports", output)(api)
let passed = 0
function test(name, fn) {
  fn()
  console.log("PASS " + name)
  passed++
}
const base = {
  model: "qwen3.5:4b",
  tools: [
    { type: "function", function: { name: "shell", parameters: { type: "object" } } },
  ],
}
test("keeps whole recent tool-call/result pairs and the latest user request", () => {
  const messages = [
    { role: "system", content: "standing instructions" },
    { role: "user", content: "old".repeat(5000) },
    { role: "assistant", content: "older answer" },
    { role: "user", content: "inspect current file" },
    {
      role: "assistant",
      tool_calls: [
        { id: "recent", type: "function", function: { name: "shell", arguments: "{}" } },
      ],
    },
    { role: "tool", tool_call_id: "recent", content: "verified current result" },
  ]
  const result = api.prepareOllamaRequest({ ...base, messages }, 12000)
  assert.equal(result.messages[0].content, "standing instructions")
  assert.match(result.messages[1].content, /older conversation turns were omitted/)
  assert.deepEqual(result.messages.slice(2), messages.slice(3))
  assert.equal(messages.length, 6)
  assert.ok(api.requestCost(result) <= 12000 - 4096 - 1024)
})
test("rejects an oversized current task without deleting the user request", () => {
  assert.throws(
    () =>
      api.prepareOllamaRequest(
        {
          ...base,
          messages: [
            { role: "system", content: "rules" },
            { role: "user", content: "x".repeat(20000) },
          ],
        },
        12000
      ),
    /not silently truncated/
  )
})
test("counts screenshots as images rather than base64 text", () => {
  const value = {
    content: [
      {
        type: "image_url",
        image_url: { url: "data:image/png;base64," + "A".repeat(300000) },
      },
    ],
  }
  assert.ok(api.requestCost(value) > 4096)
  assert.ok(api.requestCost(value) < 5000)
})
test("preserves tool definitions, existing output limit, and other model reasoning", () => {
  const value = {
    ...base,
    model: "other:4b",
    reasoning_effort: "high",
    max_tokens: 256,
    messages: [{ role: "user", content: "hi" }],
  }
  const result = api.prepareOllamaRequest(value, 32768)
  assert.equal(result.reasoning_effort, "high")
  assert.equal(result.max_tokens, 256)
  assert.deepEqual(result.tools, value.tools)
})
test("sets tested Qwen agent mode without affecting other model names", () => {
  const result = api.prepareOllamaRequest(
    { ...base, messages: [{ role: "user", content: "hi" }] },
    65536
  )
  assert.equal(result.reasoning_effort, "none")
  assert.equal(result.temperature, 0.6)
  assert.equal(
    api.prepareOllamaRequest({ ...base, model: "qwen3.5-other", messages: [] }, 65536)
      .reasoning_effort,
    undefined
  )
})
test("counts UTF-8 bytes rather than only JavaScript characters", () => {
  assert.ok(api.requestCost({ text: "你好" }) > api.requestCost({ text: "ab" }))
})
console.log(JSON.stringify({ passed }))
