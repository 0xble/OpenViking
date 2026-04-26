import { createHash } from "node:crypto"
import { readFileSync } from "node:fs"
import { homedir } from "node:os"
import { join, resolve as resolvePath } from "node:path"
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js"
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js"
import { z } from "zod"
import type { FindResult } from "./recall.js"
import {
  buildRecallResponseText,
  buildResourceRecallResponseText,
  searchMemoryScopes,
  searchResourceScope,
} from "./recall.js"

type CommitSessionResult = {
  task_id?: string
  status?: string
  memories_extracted?: Record<string, number>
  error?: unknown
}

type TaskResult = {
  status?: string
  result?: Record<string, unknown>
  error?: unknown
}

type SystemStatus = {
  user?: unknown
}

function readJson(path: string): Record<string, unknown> {
  return JSON.parse(readFileSync(path, "utf-8")) as Record<string, unknown>
}

function loadOvConf(): Record<string, unknown> {
  const defaultPath = join(homedir(), ".openviking", "ov.conf")
  const configPath = resolvePath(
    (process.env.OPENVIKING_CONFIG_FILE || defaultPath).replace(/^~/, homedir()),
  )
  try {
    return readJson(configPath)
  } catch (err) {
    const code = (err as { code?: string })?.code
    const detail = code === "ENOENT" ? `Config file not found: ${configPath}` : `Invalid config file: ${configPath}`
    process.stderr.write(`[openviking-memory] ${detail}\n`)
    process.exit(1)
  }
}

function str(value: unknown, fallback: string): string {
  if (typeof value === "string" && value.trim()) return value.trim()
  return fallback
}

function tenantStr(value: unknown, fallback: string): string {
  const resolved = str(value, fallback)
  return resolved === "default" ? "" : resolved
}

function num(value: unknown, fallback: number): number {
  if (typeof value === "number" && Number.isFinite(value)) return value
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value)
    if (Number.isFinite(parsed)) return parsed
  }
  return fallback
}

function md5Short(value: string): string {
  return createHash("md5").update(value).digest("hex").slice(0, 12)
}

function isMemoryUri(uri: string): boolean {
  return /^viking:\/\/(?:user|agent)\/[^/]+\/memories(?:\/|$)/.test(uri)
}

function totalCommitMemories(result: CommitSessionResult): number {
  return Object.values(result.memories_extracted ?? {}).reduce((sum, count) => sum + count, 0)
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

const DEFAULT_WRITE_REQUEST_TIMEOUT_MS = 120_000

function extendedWriteTimeoutMs(timeoutMs: number): number {
  return Math.max(timeoutMs, DEFAULT_WRITE_REQUEST_TIMEOUT_MS)
}

const ovConf = loadOvConf()
const serverConfig = (ovConf.server ?? {}) as Record<string, unknown>
const host = str(serverConfig.host, "127.0.0.1").replace("0.0.0.0", "127.0.0.1")
const port = Math.floor(num(serverConfig.port, 1933))

const config = {
  baseUrl: `http://${host}:${port}`,
  apiKey: str(process.env.OPENVIKING_API_KEY, str(serverConfig.root_api_key, "")),
  accountId: tenantStr(process.env.OPENVIKING_ACCOUNT, str(ovConf.default_account, "")),
  userId: tenantStr(process.env.OPENVIKING_USER, str(ovConf.default_user, "")),
  agentId: str(process.env.OPENVIKING_AGENT_ID, str(ovConf.default_agent, "codex")),
  timeoutMs: Math.max(1000, Math.floor(num(process.env.OPENVIKING_TIMEOUT_MS, 15000))),
  recallLimit: Math.max(1, Math.floor(num(process.env.OPENVIKING_RECALL_LIMIT, 6))),
  scoreThreshold: Math.min(1, Math.max(0, num(process.env.OPENVIKING_SCORE_THRESHOLD, 0.01))),
  recallResources: process.env.OPENVIKING_RECALL_RESOURCES === "1"
    || process.env.OPENVIKING_RECALL_RESOURCES === "true",
}

class OpenVikingClient {
  private runtimeIdentity: { userId: string; agentId: string } | null = null

  constructor(
    private readonly baseUrl: string,
    private readonly apiKey: string,
    private readonly accountId: string,
    private readonly userId: string,
    private readonly agentId: string,
    private readonly timeoutMs: number,
  ) {}

  private async request<T>(
    path: string,
    init: RequestInit = {},
    requestTimeoutMs = this.timeoutMs,
  ): Promise<T> {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), requestTimeoutMs)

    try {
      const headers = new Headers(init.headers ?? {})
      if (this.apiKey) headers.set("X-API-Key", this.apiKey)
      if (this.accountId) headers.set("X-OpenViking-Account", this.accountId)
      if (this.userId) headers.set("X-OpenViking-User", this.userId)
      if (this.agentId) headers.set("X-OpenViking-Agent", this.agentId)
      if (init.body && !headers.has("Content-Type")) headers.set("Content-Type", "application/json")

      const response = await fetch(`${this.baseUrl}${path}`, {
        ...init,
        headers,
        signal: controller.signal,
      })
      const payload = (await response.json().catch(() => ({}))) as {
        status?: string
        result?: T
        error?: { code?: string; message?: string }
      }

      if (!response.ok || payload.status === "error") {
        const code = payload.error?.code ? ` [${payload.error.code}]` : ""
        const message = payload.error?.message ?? `HTTP ${response.status}`
        throw new Error(`OpenViking request failed${code}: ${message}`)
      }

      return (payload.result ?? payload) as T
    } finally {
      clearTimeout(timer)
    }
  }

  async healthCheck(): Promise<boolean> {
    try {
      await this.request("/health")
      return true
    } catch {
      return false
    }
  }

  private async getRuntimeIdentity(): Promise<{ userId: string; agentId: string }> {
    if (this.runtimeIdentity) return this.runtimeIdentity

    const fallback = { userId: this.userId || "default", agentId: this.agentId || "default" }
    try {
      const status = await this.request<SystemStatus>("/api/v1/system/status")
      const userId = typeof status.user === "string" && status.user.trim() ? status.user.trim() : fallback.userId
      this.runtimeIdentity = { userId, agentId: this.agentId || "default" }
      return this.runtimeIdentity
    } catch {
      this.runtimeIdentity = fallback
      return fallback
    }
  }

  async normalizeMemoryTargetUri(targetUri: string): Promise<string> {
    const trimmed = targetUri.trim().replace(/\/+$/, "")
    const match = trimmed.match(/^viking:\/\/(user|agent)\/memories(?:\/(.*))?$/)
    if (!match) return trimmed

    const scope = match[1]
    const rest = match[2] ? `/${match[2]}` : ""
    const identity = await this.getRuntimeIdentity()
    const space = scope === "user" ? identity.userId : md5Short(`${identity.userId}:${identity.agentId}`)
    return `viking://${scope}/${space}/memories${rest}`
  }

  async find(query: string, targetUri: string, limit: number, scoreThreshold: number): Promise<FindResult> {
    const normalizedTargetUri = await this.normalizeMemoryTargetUri(targetUri)
    return this.request<FindResult>("/api/v1/search/find", {
      method: "POST",
      body: JSON.stringify({
        query,
        target_uri: normalizedTargetUri,
        limit,
        score_threshold: scoreThreshold,
      }),
    })
  }

  async read(uri: string): Promise<string> {
    return this.request<string>(`/api/v1/content/read?uri=${encodeURIComponent(uri)}`)
  }

  async createSession(): Promise<string> {
    const result = await this.request<{ session_id: string }>("/api/v1/sessions", {
      method: "POST",
      body: JSON.stringify({}),
    })
    return result.session_id
  }

  async addSessionMessage(sessionId: string, role: string, content: string): Promise<void> {
    await this.request(`/api/v1/sessions/${encodeURIComponent(sessionId)}/messages`, {
      method: "POST",
      body: JSON.stringify({ role, content }),
    })
  }

  async commitSession(sessionId: string): Promise<CommitSessionResult> {
    const result = await this.request<CommitSessionResult>(
      `/api/v1/sessions/${encodeURIComponent(sessionId)}/commit`,
      { method: "POST", body: JSON.stringify({}) },
    )

    if (!result.task_id) return result

    const deadline = Date.now() + Math.max(this.timeoutMs, 30000)
    while (Date.now() < deadline) {
      await sleep(500)
      const task = await this.getTask(result.task_id).catch(() => null)
      if (!task) break
      if (task.status === "completed") {
        const taskResult = (task.result ?? {}) as Record<string, unknown>
        return {
          ...result,
          status: "completed",
          memories_extracted: (taskResult.memories_extracted ?? {}) as Record<string, number>,
        }
      }
      if (task.status === "failed") return { ...result, status: "failed", error: task.error }
    }

    return { ...result, status: "timeout" }
  }

  async getTask(taskId: string): Promise<TaskResult> {
    return this.request<TaskResult>(`/api/v1/tasks/${encodeURIComponent(taskId)}`, { method: "GET" })
  }

  async deleteSession(sessionId: string): Promise<void> {
    await this.request(`/api/v1/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" })
  }

  async deleteUri(uri: string): Promise<void> {
    await this.request(`/api/v1/fs?uri=${encodeURIComponent(uri)}&recursive=false`, { method: "DELETE" })
  }

  async grep(
    pattern: string,
    options: { uri: string; nodeLimit?: number; levelLimit?: number; caseInsensitive?: boolean },
  ): Promise<{ matches: Array<{ uri: string; line: number; content: string }> }> {
    return this.request("/api/v1/search/grep", {
      method: "POST",
      body: JSON.stringify({
        uri: options.uri,
        pattern,
        case_insensitive: options.caseInsensitive ?? false,
        node_limit: options.nodeLimit,
        level_limit: options.levelLimit,
      }),
    })
  }

  async writeContent(
    uri: string,
    content: string,
    mode: "replace" | "append" = "replace",
  ): Promise<{ uri: string; created: boolean; mode: string; written_bytes: number }> {
    const r = await this.request<Record<string, unknown>>(
      "/api/v1/content/write",
      {
        method: "POST",
        body: JSON.stringify({ uri, content, mode, wait: true }),
      },
      extendedWriteTimeoutMs(this.timeoutMs),
    )
    return {
      uri: String(r.uri),
      created: Boolean(r.created),
      mode: String(r.mode),
      written_bytes: Number(r.written_bytes),
    }
  }
}

const client = new OpenVikingClient(
  config.baseUrl,
  config.apiKey,
  config.accountId,
  config.userId,
  config.agentId,
  config.timeoutMs,
)
const server = new McpServer({ name: "openviking-memory-codex", version: "0.1.0" })

server.tool(
  "memory_recall",
  "Search OpenViking long-term memory.",
  {
    query: z.string().describe("Search query"),
    target_uri: z.string().optional().describe("Search scope URI, default searches user and agent memories"),
    include_resources: z.boolean().optional().describe("Also search viking://resources for unscoped recall"),
    limit: z.number().optional().describe("Max results, default 6"),
    score_threshold: z.number().optional().describe("Minimum relevance score 0-1, default 0.01"),
  },
  async ({ query, target_uri, include_resources, limit, score_threshold }) => {
    const recallLimit = limit ?? config.recallLimit
    const threshold = score_threshold ?? config.scoreThreshold
    const result = await searchMemoryScopes(client, query, {
      targetUri: target_uri,
      limit: recallLimit,
      scoreThreshold: threshold,
      includeResources: include_resources ?? config.recallResources,
    })

    return { content: [{ type: "text" as const, text: buildRecallResponseText(query, result) }] }
  },
)

server.tool(
  "resource_recall",
  "Search OpenViking resources. Use when you need evidence from indexed documents, files, email, Slack, calendar, Drive, or other viking://resources content.",
  {
    query: z.string().describe("Search query"),
    target_uri: z.string().optional().describe("Resource scope URI, default viking://resources"),
    limit: z.number().optional().describe("Max results, default 6"),
    score_threshold: z.number().optional().describe("Minimum relevance score 0-1, default 0.01"),
  },
  async ({ query, target_uri, limit, score_threshold }) => {
    const recallLimit = limit ?? config.recallLimit
    const threshold = score_threshold ?? config.scoreThreshold
    const result = await searchResourceScope(client, query, {
      targetUri: target_uri,
      limit: recallLimit,
      scoreThreshold: threshold,
    })

    return { content: [{ type: "text" as const, text: buildResourceRecallResponseText(query, result) }] }
  },
)

server.tool(
  "memory_store",
  "Store information in OpenViking long-term memory.",
  {
    text: z.string().describe("Information to store"),
    role: z.string().optional().describe("Message role, default user"),
  },
  async ({ text, role }) => {
    let sessionId: string | undefined
    try {
      sessionId = await client.createSession()
      await client.addSessionMessage(sessionId, role || "user", text)
      const result = await client.commitSession(sessionId)
      const count = totalCommitMemories(result)

      if (result.status === "failed") {
        return { content: [{ type: "text" as const, text: `Memory extraction failed: ${String(result.error)}` }] }
      }
      if (result.status === "timeout") {
        return {
          content: [{
            type: "text" as const,
            text: `Memory extraction is still running (task_id=${result.task_id ?? "unknown"}).`,
          }],
        }
      }
      if (count === 0) {
        return {
          content: [{
            type: "text" as const,
            text: "Committed session, but OpenViking extracted 0 memory item(s).",
          }],
        }
      }

      return { content: [{ type: "text" as const, text: `Stored memory. Extracted ${count} item(s).` }] }
    } finally {
      if (sessionId) await client.deleteSession(sessionId).catch(() => {})
    }
  },
)

server.tool(
  "memory_write",
  "Save text verbatim at a specified memory URI and return the URI. Use for explicit 'remember this fact' saves when you already know the target URI (scope, bucket, filename). Unlike memory_store, does NOT run the extractor — content lands as-is, one file per call. Response includes the written URI so you can verify or reference it downstream without guessing.",
  {
    uri: z
      .string()
      .describe(
        "Memory URI to write (e.g. viking://user/<id>/memories/preferences/mem_foo.md or viking://agent/<id>/memories/profile.md).",
      ),
    content: z.string().describe("Content to store verbatim"),
    mode: z.enum(["replace", "append"]).optional().describe("replace (default) or append"),
  },
  async ({ uri, content, mode }) => {
    if (!isMemoryUri(uri)) {
      return { content: [{ type: "text" as const, text: `Refusing to write non-memory URI: ${uri}` }] }
    }
    const result = await client.writeContent(uri, content, mode ?? "replace")
    const verb = result.created ? "created" : "updated"
    return { content: [{ type: "text" as const, text: `${verb} ${result.uri}` }] }
  },
)

server.tool(
  "memory_forget",
  "Delete an exact OpenViking memory URI. Use memory_recall first if you only have a query.",
  {
    uri: z.string().describe("Exact memory URI to delete"),
  },
  async ({ uri }) => {
    if (!isMemoryUri(uri)) {
      return { content: [{ type: "text" as const, text: `Refusing to delete non-memory URI: ${uri}` }] }
    }

    await client.deleteUri(uri)
    return { content: [{ type: "text" as const, text: `Deleted memory: ${uri}` }] }
  },
)

const SESSION_RECALL_NODE_LIMIT = 300
const SESSION_RECALL_LEVEL_LIMIT = 5
const SESSION_RECALL_SNIPPET_WIDTH = 200

function extractSessionId(uri: string): string | null {
  const match = uri.match(/^viking:\/\/session\/([^/]+)/)
  return match ? match[1]! : null
}

function snippetFromContent(content: string, pattern: string, width = SESSION_RECALL_SNIPPET_WIDTH): string {
  const lower = content.toLowerCase()
  const idx = lower.indexOf(pattern.toLowerCase())
  if (idx < 0) return content.slice(0, width)
  const start = Math.max(0, idx - Math.floor(width / 3))
  const end = Math.min(content.length, start + width)
  const prefix = start > 0 ? "..." : ""
  const suffix = end < content.length ? "..." : ""
  return `${prefix}${content.slice(start, end).replace(/\s+/g, " ").trim()}${suffix}`
}

server.tool(
  "session_recall",
  "Find prior sessions whose transcript contains the given query. Substring (not semantic) match over session message bodies. Use when you need to recall an earlier conversation where a specific string, error, or topic was discussed.",
  {
    query: z.string().describe("Literal text to grep across session transcripts"),
    limit: z.number().optional().describe("Max sessions to return (default: 10)"),
    scope_uri: z.string().optional().describe("Session scope URI (default: viking://session)"),
    case_insensitive: z.boolean().optional().describe("Case-insensitive match (default: false)"),
  },
  async ({ query, limit, scope_uri, case_insensitive }) => {
    const cap = Math.max(1, Math.min(50, limit ?? 10))
    const scope = scope_uri ?? "viking://session"
    const { matches } = await client.grep(query, {
      uri: scope,
      nodeLimit: SESSION_RECALL_NODE_LIMIT,
      levelLimit: SESSION_RECALL_LEVEL_LIMIT,
      caseInsensitive: case_insensitive ?? false,
    })

    if (matches.length === 0) {
      return { content: [{ type: "text" as const, text: `No sessions matched "${query}" under ${scope}.` }] }
    }

    const bySession = new Map<string, { count: number; firstSnippet: string; firstUri: string }>()
    for (const m of matches) {
      const sid = extractSessionId(m.uri)
      if (!sid) continue
      const entry = bySession.get(sid)
      if (entry) {
        entry.count += 1
      } else {
        bySession.set(sid, {
          count: 1,
          firstSnippet: snippetFromContent(m.content, query),
          firstUri: m.uri,
        })
      }
    }

    if (bySession.size === 0) {
      return { content: [{ type: "text" as const, text: `Matches found but none mapped to a session id under ${scope}.` }] }
    }

    const truncated = matches.length >= SESSION_RECALL_NODE_LIMIT
    const top = [...bySession.entries()].sort((a, b) => b[1].count - a[1].count).slice(0, cap)
    const lines = top.map(([sid, info]) =>
      `- ${sid} (${info.count} match${info.count === 1 ? "" : "es"})\n    ${info.firstSnippet}`,
    )
    const note = truncated
      ? `\n\n(truncated at ${SESSION_RECALL_NODE_LIMIT} raw matches; refine query for full coverage)`
      : ""

    return {
      content: [{
        type: "text" as const,
        text: `Found ${bySession.size} session${bySession.size === 1 ? "" : "s"} matching "${query}"${bySession.size > cap ? `, showing ${cap}` : ""}:\n\n${lines.join("\n")}${note}`,
      }],
    }
  },
)

server.tool(
  "memory_health",
  "Check whether the OpenViking server is reachable.",
  {},
  async () => {
    const ok = await client.healthCheck()
    const text = ok
      ? `OpenViking is reachable at ${config.baseUrl}.`
      : `OpenViking is unreachable at ${config.baseUrl}.`
    return { content: [{ type: "text" as const, text }] }
  },
)

const transport = new StdioServerTransport()
await server.connect(transport)
