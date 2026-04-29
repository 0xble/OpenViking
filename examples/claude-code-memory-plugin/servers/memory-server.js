/**
 * OpenViking Memory MCP Server for Claude Code
 *
 * Exposes OpenViking long-term memory as MCP tools:
 *   - memory_recall  : semantic search across memories
 *   - memory_store   : extract and persist new memories
 *   - memory_forget  : delete memories by URI or query
 *   - session_recall : substring search across prior session transcripts
 *
 * Ported from the OpenClaw context-engine plugin (openclaw-plugin/).
 * Adapted for Claude Code's MCP server interface (stdio transport).
 */
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { createHash } from "node:crypto";
import { clampScore, formatOrderSourceNote, formatScopeFailures, searchMemoryScopes, searchResourceScope, } from "./recall.js";
// ---------------------------------------------------------------------------
// Configuration — loaded from the Claude Code client config.
// Env var: OPENVIKING_CC_CONFIG_FILE
// Default: ~/.openviking/claude-code-memory-plugin/config.json
//
// In local mode, apiKey defaults to the local OpenViking server config:
// OPENVIKING_CONFIG_FILE or ~/.openviking/ov.conf
// ---------------------------------------------------------------------------
import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join, resolve as resolvePath } from "node:path";
const DEFAULT_CLIENT_CONFIG_PATH = join(homedir(), ".openviking", "claude-code-memory-plugin", "config.json");
const DEFAULT_SERVER_CONFIG_PATH = join(homedir(), ".openviking", "ov.conf");
function fatal(message) {
    process.stderr.write(`[openviking-memory] ${message}\n`);
    process.exit(1);
}
function resolveEnvVars(value) {
    return value.replace(/\$\{([^}]+)\}/g, (_, envVar) => {
        const envValue = process.env[envVar];
        if (typeof envValue !== "string" || envValue === "") {
            fatal(`Environment variable ${envVar} is not set`);
        }
        return envValue;
    });
}
function resolveString(value, fallback) {
    if (typeof value === "string" && value.trim())
        return resolveEnvVars(value.trim());
    return fallback;
}
function resolveTenantString(value, fallback) {
    const resolved = resolveString(value, fallback);
    return resolved === "default" ? "" : resolved;
}
function resolveConfigPath(rawValue, fallback) {
    return resolvePath(resolveString(rawValue, fallback).replace(/^~/, homedir()));
}
function normalizeBaseUrl(value) {
    return resolveString(value, "").replace(/\/+$/, "");
}
function requireBaseUrl(value) {
    const resolved = normalizeBaseUrl(value);
    if (!resolved) {
        fatal("Claude Code client config: baseUrl is required when mode is \"remote\"");
    }
    return resolved;
}
function clampPort(value) {
    return Math.max(1, Math.min(65535, Math.floor(num(value, 1933))));
}
function loadRequiredJson(configPath, label) {
    let raw;
    try {
        raw = readFileSync(configPath, "utf-8");
    }
    catch (err) {
        const code = err?.code;
        const msg = code === "ENOENT"
            ? `${label} not found: ${configPath}\n  Create it and set at least: { "mode": "local" }`
            : `Failed to read ${label}: ${configPath} — ${err?.message || String(err)}`;
        fatal(msg);
    }
    try {
        const parsed = JSON.parse(raw);
        if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
            fatal(`${label} must contain a JSON object: ${configPath}`);
        }
        return parsed;
    }
    catch (err) {
        fatal(`Invalid JSON in ${configPath}: ${err?.message || String(err)}`);
    }
}
function loadOptionalJson(configPath) {
    try {
        const parsed = JSON.parse(readFileSync(configPath, "utf-8"));
        if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
            return { file: null, error: `JSON root must be an object: ${configPath}` };
        }
        return { file: parsed, error: null };
    }
    catch (err) {
        if (err?.code === "ENOENT") {
            return { file: null, error: null };
        }
        return { file: null, error: err?.message || String(err) };
    }
}
function num(val, fallback) {
    if (typeof val === "number" && Number.isFinite(val))
        return val;
    if (typeof val === "string" && val.trim()) {
        const n = Number(val);
        if (Number.isFinite(n))
            return n;
    }
    return fallback;
}
const clientConfigPath = resolveConfigPath(process.env.OPENVIKING_CC_CONFIG_FILE, DEFAULT_CLIENT_CONFIG_PATH);
const clientFile = loadRequiredJson(clientConfigPath, "Claude Code client config");
const mode = clientFile.mode === "remote" ? "remote" : "local";
const serverConfigPath = resolveConfigPath(process.env.OPENVIKING_CONFIG_FILE, DEFAULT_SERVER_CONFIG_PATH);
const serverConfigResult = mode === "local"
    ? loadOptionalJson(serverConfigPath)
    : { file: null, error: null };
const serverCfg = (serverConfigResult.file?.server ?? {});
const serverFile = serverConfigResult.file ?? {};
const config = {
    mode,
    configPath: clientConfigPath,
    serverConfigPath,
    serverConfigError: serverConfigResult.error,
    baseUrl: mode === "remote"
        ? requireBaseUrl(clientFile.baseUrl)
        : `http://127.0.0.1:${clampPort(serverCfg.port)}`,
    apiKey: resolveString(clientFile.apiKey, "") || (mode === "local" ? resolveString(serverCfg.root_api_key, "") : ""),
    accountId: resolveTenantString(clientFile.account ?? clientFile.accountId, resolveString(serverFile.default_account, "")),
    userId: resolveTenantString(clientFile.user ?? clientFile.userId, resolveString(serverFile.default_user, "")),
    agentId: resolveString(clientFile.agentId, "claude-code"),
    timeoutMs: Math.max(1000, Math.floor(num(clientFile.timeoutMs, 15000))),
    recallLimit: Math.max(1, Math.floor(num(clientFile.recallLimit, 6))),
    scoreThreshold: Math.min(1, Math.max(0, num(clientFile.scoreThreshold, 0.01))),
    recallResources: clientFile.recallResources === true,
    recallReadLineLimit: Math.max(1, Math.floor(num(clientFile.recallReadLineLimit, 120))),
    recallContentMaxChars: Math.max(200, Math.floor(num(clientFile.recallContentMaxChars, 4000))),
};
// ---------------------------------------------------------------------------
// OpenViking HTTP Client (ported from openclaw-plugin/client.ts)
// ---------------------------------------------------------------------------
const MEMORY_URI_PATTERNS = [
    /^viking:\/\/user\/(?:[^/]+\/)?memories(?:\/|$)/,
    /^viking:\/\/agent\/(?:[^/]+\/)?memories(?:\/|$)/,
];
const USER_STRUCTURE_DIRS = new Set(["memories"]);
const AGENT_STRUCTURE_DIRS = new Set(["memories", "skills", "instructions", "workspaces"]);
function md5Short(input) {
    return createHash("md5").update(input).digest("hex").slice(0, 12);
}
function isMemoryUri(uri) {
    return MEMORY_URI_PATTERNS.some((p) => p.test(uri));
}
class OpenVikingClient {
    baseUrl;
    apiKey;
    accountId;
    userId;
    agentId;
    timeoutMs;
    resolvedSpaceByScope = {};
    runtimeIdentity = null;
    constructor(baseUrl, apiKey, accountId, userId, agentId, timeoutMs) {
        this.baseUrl = baseUrl;
        this.apiKey = apiKey;
        this.accountId = accountId;
        this.userId = userId;
        this.agentId = agentId;
        this.timeoutMs = timeoutMs;
    }
    async request(path, init = {}, requestTimeoutMs = this.timeoutMs) {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), requestTimeoutMs);
        try {
            const headers = new Headers(init.headers ?? {});
            if (this.apiKey)
                headers.set("X-API-Key", this.apiKey);
            if (this.accountId)
                headers.set("X-OpenViking-Account", this.accountId);
            if (this.userId)
                headers.set("X-OpenViking-User", this.userId);
            if (this.agentId)
                headers.set("X-OpenViking-Agent", this.agentId);
            if (init.body && !headers.has("Content-Type"))
                headers.set("Content-Type", "application/json");
            const response = await fetch(`${this.baseUrl}${path}`, {
                ...init,
                headers,
                signal: controller.signal,
            });
            const payload = (await response.json().catch(() => ({})));
            if (!response.ok || payload.status === "error") {
                const code = payload.error?.code ? ` [${payload.error.code}]` : "";
                const message = payload.error?.message ?? `HTTP ${response.status}`;
                throw new Error(`OpenViking request failed${code}: ${message}`);
            }
            return (payload.result ?? payload);
        }
        finally {
            clearTimeout(timer);
        }
    }
    async healthCheck() {
        try {
            await this.request("/health");
            return true;
        }
        catch {
            return false;
        }
    }
    async ls(uri) {
        return this.request(`/api/v1/fs/ls?uri=${encodeURIComponent(uri)}&output=original`);
    }
    async getRuntimeIdentity() {
        if (this.runtimeIdentity)
            return this.runtimeIdentity;
        const fallback = { userId: this.userId || "default", agentId: this.agentId || "default" };
        try {
            const status = await this.request("/api/v1/system/status");
            const userId = typeof status.user === "string" && status.user.trim() ? status.user.trim() : fallback.userId;
            this.runtimeIdentity = { userId, agentId: this.agentId || "default" };
            return this.runtimeIdentity;
        }
        catch {
            this.runtimeIdentity = fallback;
            return fallback;
        }
    }
    async resolveScopeSpace(scope) {
        const cached = this.resolvedSpaceByScope[scope];
        if (cached)
            return cached;
        const identity = await this.getRuntimeIdentity();
        const fallbackSpace = scope === "user" ? identity.userId : md5Short(`${identity.userId}:${identity.agentId}`);
        const reservedDirs = scope === "user" ? USER_STRUCTURE_DIRS : AGENT_STRUCTURE_DIRS;
        try {
            const entries = await this.ls(`viking://${scope}`);
            const spaces = entries
                .filter((e) => e?.isDir === true)
                .map((e) => (typeof e.name === "string" ? e.name.trim() : ""))
                .filter((n) => n && !n.startsWith(".") && !reservedDirs.has(n));
            if (spaces.length > 0) {
                if (spaces.includes(fallbackSpace)) {
                    this.resolvedSpaceByScope[scope] = fallbackSpace;
                    return fallbackSpace;
                }
                if (scope === "user" && spaces.includes("default")) {
                    this.resolvedSpaceByScope[scope] = "default";
                    return "default";
                }
                if (spaces.length === 1) {
                    this.resolvedSpaceByScope[scope] = spaces[0];
                    return spaces[0];
                }
            }
        }
        catch { /* fall through */ }
        this.resolvedSpaceByScope[scope] = fallbackSpace;
        return fallbackSpace;
    }
    async normalizeTargetUri(targetUri) {
        const trimmed = targetUri.trim().replace(/\/+$/, "");
        const match = trimmed.match(/^viking:\/\/(user|agent)(?:\/(.*))?$/);
        if (!match)
            return trimmed;
        const scope = match[1];
        const rawRest = (match[2] ?? "").trim();
        if (!rawRest)
            return trimmed;
        const parts = rawRest.split("/").filter(Boolean);
        if (parts.length === 0)
            return trimmed;
        const reservedDirs = scope === "user" ? USER_STRUCTURE_DIRS : AGENT_STRUCTURE_DIRS;
        if (!reservedDirs.has(parts[0]))
            return trimmed;
        const space = await this.resolveScopeSpace(scope);
        return `viking://${scope}/${space}/${parts.join("/")}`;
    }
    async find(query, options) {
        const normalizedTargetUri = await this.normalizeTargetUri(options.targetUri);
        return this.request("/api/v1/search/find", {
            method: "POST",
            body: JSON.stringify({
                query,
                target_uri: normalizedTargetUri,
                limit: options.limit,
                score_threshold: options.scoreThreshold,
            }),
        });
    }
    async read(uri, limit = -1) {
        return this.request(`/api/v1/content/read?uri=${encodeURIComponent(uri)}&limit=${encodeURIComponent(String(limit))}`);
    }
    async createSession() {
        const result = await this.request("/api/v1/sessions", {
            method: "POST",
            body: JSON.stringify({}),
        });
        return result.session_id;
    }
    async addSessionMessage(sessionId, role, content) {
        await this.request(`/api/v1/sessions/${encodeURIComponent(sessionId)}/messages`, {
            method: "POST",
            body: JSON.stringify({ role, content }),
        });
    }
    async commitSession(sessionId) {
        const result = await this.request(`/api/v1/sessions/${encodeURIComponent(sessionId)}/commit`, { method: "POST", body: JSON.stringify({}) });
        if (!result.task_id)
            return result;
        const deadline = Date.now() + Math.max(this.timeoutMs, 30000);
        while (Date.now() < deadline) {
            await sleep(500);
            const task = await this.getTask(result.task_id).catch(() => null);
            if (!task)
                break;
            if (task.status === "completed") {
                const taskResult = (task.result ?? {});
                return {
                    ...result,
                    status: "completed",
                    memories_extracted: (taskResult.memories_extracted ?? {}),
                };
            }
            if (task.status === "failed")
                return { ...result, status: "failed", error: task.error };
        }
        return { ...result, status: "timeout" };
    }
    async getTask(taskId) {
        return this.request(`/api/v1/tasks/${encodeURIComponent(taskId)}`, { method: "GET" });
    }
    async deleteSession(sessionId) {
        await this.request(`/api/v1/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
    }
    async deleteUri(uri) {
        await this.request(`/api/v1/fs?uri=${encodeURIComponent(uri)}&recursive=false`, {
            method: "DELETE",
        });
    }
    async grep(pattern, options) {
        return this.request("/api/v1/search/grep", {
            method: "POST",
            body: JSON.stringify({
                uri: options.uri,
                pattern,
                case_insensitive: options.caseInsensitive ?? false,
                node_limit: options.nodeLimit,
                level_limit: options.levelLimit,
            }),
        });
    }
    /**
     * Write verbatim content to a memory URI via POST /api/v1/content/write.
     * Creates the file (and missing parent dirs) when it does not yet exist.
     */
    async writeContent(uri, content, mode = "replace") {
        return this.writeContentOnce(uri, content, mode);
    }
    async writeContentOnce(uri, content, mode) {
        const r = await this.request("/api/v1/content/write", {
            method: "POST",
            body: JSON.stringify({ uri, content, mode, wait: false }),
        });
        return {
            uri: String(r.uri),
            created: Boolean(r.created),
            mode: String(r.mode),
            written_bytes: Number(r.written_bytes),
        };
    }
}
// ---------------------------------------------------------------------------
// Memory ranking helpers (ported from openclaw-plugin/memory-ranking.ts)
// ---------------------------------------------------------------------------
function normalizeDedupeText(text) {
    return text.toLowerCase().replace(/\s+/g, " ").trim();
}
function getMemoryDedupeKey(item) {
    const abstract = normalizeDedupeText(item.abstract ?? item.overview ?? "");
    const category = (item.category ?? "").toLowerCase() || "unknown";
    if (abstract)
        return `abstract:${category}:${abstract}`;
    return `uri:${item.uri}`;
}
function postProcessMemories(items, options) {
    const deduped = [];
    const seen = new Set();
    const sorted = [...items].sort((a, b) => clampScore(b.score) - clampScore(a.score));
    for (const item of sorted) {
        if (options.leafOnly && item.level !== 2)
            continue;
        if (clampScore(item.score) < options.scoreThreshold)
            continue;
        const key = getMemoryDedupeKey(item);
        if (seen.has(key))
            continue;
        seen.add(key);
        deduped.push(item);
        if (deduped.length >= options.limit)
            break;
    }
    return deduped;
}
function formatMemoryLines(items) {
    return items
        .map((item, i) => {
        const score = clampScore(item.score);
        const abstract = item.abstract?.trim() || item.overview?.trim() || item.uri;
        const category = item.category ?? "memory";
        return `${i + 1}. [${category}] ${abstract} (${(score * 100).toFixed(0)}%)`;
    })
        .join("\n");
}
function truncateContent(content, maxChars) {
    const trimmed = content.trim();
    if (trimmed.length <= maxChars)
        return trimmed;
    return `${trimmed.slice(0, maxChars).trimEnd()}\n[truncated to ${maxChars} characters]`;
}
async function formatRecallItemContent(item) {
    const category = item.category ?? item.context_type ?? "memory";
    if (item.level === 2) {
        try {
            const content = await client.read(item.uri, config.recallReadLineLimit);
            if (content?.trim())
                return `- [${category}] ${truncateContent(content, config.recallContentMaxChars)}`;
        }
        catch { /* fallback */ }
    }
    return `- [${category}] ${item.abstract ?? item.overview ?? item.uri}`;
}
// Query-aware ranking (ported from openclaw-plugin/memory-ranking.ts)
const PREFERENCE_QUERY_RE = /prefer|preference|favorite|favourite|like|偏好|喜欢|爱好|更倾向/i;
const TEMPORAL_QUERY_RE = /when|what time|date|day|month|year|yesterday|today|tomorrow|last|next|什么时候|何时|哪天|几月|几年|昨天|今天|明天/i;
const QUERY_TOKEN_RE = /[a-z0-9]{2,}/gi;
const QUERY_TOKEN_STOPWORDS = new Set([
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how", "did", "does",
    "is", "are", "was", "were", "the", "and", "for", "with", "from", "that", "this", "your", "you",
]);
function buildQueryProfile(query) {
    const text = query.trim();
    const allTokens = text.toLowerCase().match(QUERY_TOKEN_RE) ?? [];
    const tokens = allTokens.filter((t) => !QUERY_TOKEN_STOPWORDS.has(t));
    return {
        tokens,
        wantsPreference: PREFERENCE_QUERY_RE.test(text),
        wantsTemporal: TEMPORAL_QUERY_RE.test(text),
    };
}
function lexicalOverlapBoost(tokens, text) {
    if (tokens.length === 0 || !text)
        return 0;
    const haystack = ` ${text.toLowerCase()} `;
    let matched = 0;
    for (const token of tokens.slice(0, 8)) {
        if (haystack.includes(token))
            matched += 1;
    }
    return Math.min(0.2, (matched / Math.min(tokens.length, 4)) * 0.2);
}
function rankForInjection(item, query) {
    const baseScore = clampScore(item.score);
    const abstract = (item.abstract ?? item.overview ?? "").trim();
    const leafBoost = item.level === 2 ? 0.12 : 0;
    const cat = (item.category ?? "").toLowerCase();
    const eventBoost = query.wantsTemporal && (cat === "events" || item.uri.includes("/events/")) ? 0.1 : 0;
    const prefBoost = query.wantsPreference && (cat === "preferences" || item.uri.includes("/preferences/")) ? 0.08 : 0;
    const overlapBoost = lexicalOverlapBoost(query.tokens, `${item.uri} ${abstract}`);
    return baseScore + leafBoost + eventBoost + prefBoost + overlapBoost;
}
function pickMemoriesForInjection(items, limit, queryText) {
    if (items.length === 0 || limit <= 0)
        return [];
    const query = buildQueryProfile(queryText);
    const sorted = [...items].sort((a, b) => rankForInjection(b, query) - rankForInjection(a, query));
    const deduped = [];
    const seen = new Set();
    for (const item of sorted) {
        const key = (item.abstract ?? item.overview ?? "").trim().toLowerCase() || item.uri;
        if (seen.has(key))
            continue;
        seen.add(key);
        deduped.push(item);
    }
    const leaves = deduped.filter((item) => item.level === 2);
    if (leaves.length >= limit)
        return leaves.slice(0, limit);
    const picked = [...leaves];
    const used = new Set(leaves.map((item) => item.uri));
    for (const item of deduped) {
        if (picked.length >= limit)
            break;
        if (used.has(item.uri))
            continue;
        picked.push(item);
    }
    return picked;
}
// ---------------------------------------------------------------------------
// Shared search helpers
// ---------------------------------------------------------------------------
function totalCommitMemories(result) {
    return Object.values(result.memories_extracted ?? {}).reduce((sum, count) => sum + count, 0);
}
const MEMORY_BODY_MAX_CHARS = 500;
function singularize(plural) {
    if (plural.endsWith("ies"))
        return plural.replace(/ies$/, "y");
    if (plural.endsWith("s"))
        return plural.replace(/s$/, "");
    return plural;
}
function humanBytes(n) {
    if (n < 1024)
        return `${n} B`;
    return `${(n / 1024).toFixed(1)} KB`;
}
function bucketSingularFromUri(uri) {
    const m = uri.match(/\/memories\/([^/]+)/);
    if (!m)
        return "memory";
    const seg = m[1].replace(/\.md$/, "");
    if (seg === "profile")
        return "profile";
    return singularize(seg);
}
function prettyTitleFromUri(uri) {
    const stem = uri.split("/").pop()?.replace(/\.md$/, "") ?? "";
    if (!stem || stem.startsWith("mem_"))
        return "(unnamed)";
    return stem
        .replace(/[_-]+/g, " ")
        .split(" ")
        .filter(Boolean)
        .map((w) => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase())
        .join(" ");
}
function quoteBody(text, max) {
    if (!text)
        return "";
    const truncated = text.length > max;
    const shown = truncated ? text.slice(0, max).trimEnd() : text;
    const lines = shown.split("\n").map((l) => `> ${l}`).join("\n");
    if (truncated) {
        return `${lines}\n[showing first ${max} of ${text.length.toLocaleString()} chars]`;
    }
    return lines;
}
function formatMemoryWriteMessage(opts) {
    const verb = opts.created ? "Created" : opts.mode === "append" ? "Appended to" : "Updated";
    const bucket = bucketSingularFromUri(opts.uri);
    const title = prettyTitleFromUri(opts.uri);
    const sizeStr = humanBytes(opts.writtenBytes);
    const titlePart = title === "(unnamed)" ? "(unnamed)" : `"${title}"`;
    const header = bucket === "profile"
        ? `${verb} profile memory (${sizeStr}).`
        : `${verb} ${bucket} memory ${titlePart} (${sizeStr}).`;
    const body = quoteBody(opts.content, MEMORY_BODY_MAX_CHARS);
    return body ? `${header}\n${opts.uri}\n\n${body}` : `${header}\n${opts.uri}`;
}
function formatMemoryForgetMessage(uri, score) {
    const bucket = bucketSingularFromUri(uri);
    const title = prettyTitleFromUri(uri);
    const scoreSuffix = typeof score === "number" ? ` (matched at ${(score * 100).toFixed(0)}%)` : "";
    const titlePart = title === "(unnamed)" ? "(unnamed)" : `"${title}"`;
    const header = bucket === "profile"
        ? `Forgot profile memory${scoreSuffix}.`
        : `Forgot ${bucket} memory ${titlePart}${scoreSuffix}.`;
    return `${header}\n${uri}`;
}
function formatMemoryStoreMessage(opts) {
    const cats = opts.memoriesByCategory ?? {};
    const total = Object.values(cats).reduce((sum, n) => sum + (n ?? 0), 0);
    const breakdown = Object.entries(cats)
        .filter(([, n]) => (n ?? 0) > 0)
        .map(([cat, n]) => `${n} ${n === 1 ? singularize(cat) : cat}`)
        .join(", ");
    const header = total === 0
        ? "Stored input but extracted 0 memories."
        : `Extracted ${total} ${total === 1 ? "memory" : "memories"}${breakdown ? ` (${breakdown})` : ""}.`;
    const body = quoteBody(opts.text ?? "", MEMORY_BODY_MAX_CHARS);
    return body ? `${header}\n\n${body}` : header;
}
function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}
// ---------------------------------------------------------------------------
// MCP Server
// ---------------------------------------------------------------------------
const client = new OpenVikingClient(config.baseUrl, config.apiKey, config.accountId, config.userId, config.agentId, config.timeoutMs);
const server = new McpServer({
    name: "openviking-memory",
    version: "0.1.0",
});
// -- Tool: memory_recall --------------------------------------------------
server.tool("memory_recall", "Search long-term memories from OpenViking. Use when you need past user preferences, facts, decisions, or any previously stored information.", {
    query: z.string().describe("Search query — describe what you want to recall"),
    limit: z.number().optional().describe("Max results to return (default: 6)"),
    score_threshold: z.number().optional().describe("Min relevance score 0-1 (default: 0.01)"),
    target_uri: z.string().optional().describe("Search scope URI, e.g. viking://user/memories"),
    include_resources: z.boolean().optional().describe("Also search viking://resources for unscoped recall"),
}, async ({ query, limit, score_threshold, target_uri, include_resources }) => {
    const recallLimit = limit ?? config.recallLimit;
    const threshold = score_threshold ?? config.scoreThreshold;
    const candidateLimit = Math.max(recallLimit * 4, 20);
    const searchResult = await searchMemoryScopes(client, query, {
        targetUri: target_uri,
        limit: candidateLimit,
        scoreThreshold: threshold,
        includeResources: include_resources ?? config.recallResources,
    });
    const processed = postProcessMemories(searchResult.memories, { limit: candidateLimit, scoreThreshold: threshold });
    const processedResources = postProcessMemories(searchResult.resources, { limit: candidateLimit, scoreThreshold: threshold });
    const memories = pickMemoriesForInjection(processed, recallLimit, query);
    const resources = pickMemoriesForInjection(processedResources, recallLimit, query);
    const recallItems = [...memories, ...resources].slice(0, recallLimit);
    const notes = [
        formatScopeFailures(searchResult.failedScopes),
        formatOrderSourceNote(query),
    ].filter(Boolean);
    if (recallItems.length === 0) {
        return {
            content: [{
                    type: "text",
                    text: ["No relevant OpenViking context found.", ...notes].join("\n\n"),
                }],
        };
    }
    const lines = await Promise.all(recallItems.map(formatRecallItemContent));
    return {
        content: [{
                type: "text",
                text: [
                    `Found ${recallItems.length} relevant OpenViking item(s):\n\n${lines.join("\n")}\n\n---\n${formatMemoryLines(recallItems)}`,
                    ...notes,
                ].join("\n\n"),
            }],
    };
});
server.tool("resource_recall", "Search OpenViking resources. Use when you need evidence from indexed documents, files, email, Slack, calendar, Drive, or other viking://resources content.", {
    query: z.string().describe("Search query — describe the resource evidence you want to find"),
    limit: z.number().optional().describe("Max results to return (default: 6)"),
    score_threshold: z.number().optional().describe("Min relevance score 0-1 (default: 0.01)"),
    target_uri: z.string().optional().describe("Resource scope URI, e.g. viking://resources/email"),
}, async ({ query, limit, score_threshold, target_uri }) => {
    const recallLimit = limit ?? config.recallLimit;
    const threshold = score_threshold ?? config.scoreThreshold;
    const candidateLimit = Math.max(recallLimit * 4, 20);
    const searchResult = await searchResourceScope(client, query, {
        targetUri: target_uri,
        limit: candidateLimit,
        scoreThreshold: threshold,
    });
    const processedResources = postProcessMemories(searchResult.resources, { limit: candidateLimit, scoreThreshold: threshold });
    const resources = pickMemoriesForInjection(processedResources, recallLimit, query);
    if (resources.length === 0) {
        return {
            content: [{
                    type: "text",
                    text: `No relevant OpenViking resources found for "${query}".`,
                }],
        };
    }
    const lines = await Promise.all(resources.map(formatRecallItemContent));
    return {
        content: [{
                type: "text",
                text: `Found ${resources.length} relevant OpenViking resource(s):\n\n${lines.join("\n")}\n\n---\n${formatMemoryLines(resources)}`,
            }],
    };
});
// -- Tool: memory_store ---------------------------------------------------
server.tool("memory_store", "Store information into OpenViking long-term memory. Use when the user says 'remember this', shares preferences, important facts, decisions, or any information worth persisting across sessions.", {
    text: z.string().describe("The information to store as memory"),
    role: z.string().optional().describe("Message role: 'user' (default) or 'assistant'"),
}, async ({ text, role }) => {
    const msgRole = role || "user";
    let sessionId;
    try {
        sessionId = await client.createSession();
        await client.addSessionMessage(sessionId, msgRole, text);
        const result = await client.commitSession(sessionId);
        const count = totalCommitMemories(result);
        if (result.status === "failed") {
            return {
                content: [{
                        type: "text",
                        text: `Memory extraction failed: ${String(result.error)}`,
                    }],
            };
        }
        if (result.status === "timeout") {
            return {
                content: [{
                        type: "text",
                        text: `Memory extraction is still running (task_id=${result.task_id ?? "unknown"}).`,
                    }],
            };
        }
        const storeMessage = formatMemoryStoreMessage({
            text,
            memoriesByCategory: result.memories_extracted,
        });
        return {
            content: [{
                    type: "text",
                    text: storeMessage,
                }],
        };
    }
    finally {
        if (sessionId) {
            await client.deleteSession(sessionId).catch(() => { });
        }
    }
});
// -- Tool: memory_write ---------------------------------------------------
server.tool("memory_write", "Save text verbatim at a specified memory URI and return the URI. Use for explicit 'remember this fact' saves when you already know the target URI (scope, bucket, filename). Unlike memory_store, does NOT run the extractor — content lands as-is, one file per call. Response includes the written URI so you can verify or reference it downstream without guessing.", {
    uri: z
        .string()
        .describe("Memory URI to write (e.g. viking://user/<id>/memories/preferences/mem_foo.md or viking://agent/<id>/memories/profile.md)."),
    content: z.string().describe("Content to store verbatim"),
    mode: z
        .enum(["replace", "append"])
        .optional()
        .describe("replace (default) or append"),
}, async ({ uri, content, mode }) => {
    if (!isMemoryUri(uri)) {
        return {
            content: [
                {
                    type: "text",
                    text: `Refusing to write non-memory URI: ${uri}`,
                },
            ],
        };
    }
    const result = await client.writeContent(uri, content, mode ?? "replace");
    const message = formatMemoryWriteMessage({
        uri: result.uri,
        content,
        mode: result.mode,
        created: result.created ?? false,
        writtenBytes: result.written_bytes,
    });
    return {
        content: [
            {
                type: "text",
                text: message,
            },
        ],
    };
});
// -- Tool: memory_forget --------------------------------------------------
server.tool("memory_forget", "Delete a memory from OpenViking. Provide an exact URI for direct deletion, or a search query to find and delete matching memories.", {
    uri: z.string().optional().describe("Exact viking:// memory URI to delete"),
    query: z.string().optional().describe("Search query to find the memory to delete"),
    target_uri: z.string().optional().describe("Search scope URI (default: viking://user/memories)"),
}, async ({ uri, query, target_uri }) => {
    // Direct URI deletion
    if (uri) {
        if (!isMemoryUri(uri)) {
            return { content: [{ type: "text", text: `Refusing to delete non-memory URI: ${uri}` }] };
        }
        await client.deleteUri(uri);
        return { content: [{ type: "text", text: formatMemoryForgetMessage(uri) }] };
    }
    if (!query) {
        return { content: [{ type: "text", text: "Please provide either a uri or query parameter." }] };
    }
    // Search then delete
    const candidateLimit = 20;
    let candidates;
    if (target_uri) {
        const result = await client.find(query, { targetUri: target_uri, limit: candidateLimit, scoreThreshold: 0 });
        candidates = postProcessMemories(result.memories ?? [], {
            limit: candidateLimit,
            scoreThreshold: config.scoreThreshold,
            leafOnly: true,
        }).filter((item) => isMemoryUri(item.uri));
    }
    else {
        const { memories: leafMemories } = await searchMemoryScopes(client, query, {
            limit: candidateLimit,
            scoreThreshold: config.scoreThreshold,
        });
        candidates = postProcessMemories(leafMemories, {
            limit: candidateLimit,
            scoreThreshold: config.scoreThreshold,
            leafOnly: true,
        }).filter((item) => isMemoryUri(item.uri));
    }
    if (candidates.length === 0) {
        return { content: [{ type: "text", text: "No matching memories found. Try a more specific query." }] };
    }
    // Auto-delete if single strong match
    const top = candidates[0];
    if (candidates.length === 1 && clampScore(top.score) >= 0.85) {
        await client.deleteUri(top.uri);
        return {
            content: [
                {
                    type: "text",
                    text: formatMemoryForgetMessage(top.uri, top.score ?? undefined),
                },
            ],
        };
    }
    // List candidates for confirmation
    const list = candidates
        .map((item) => `- ${item.uri} — ${item.abstract?.trim() || "?"} (${(clampScore(item.score) * 100).toFixed(0)}%)`)
        .join("\n");
    return {
        content: [{
                type: "text",
                text: `Found ${candidates.length} candidate memories. Please specify the exact URI to delete:\n\n${list}`,
            }],
    };
});
// -- Tool: session_recall -------------------------------------------------
const SESSION_RECALL_NODE_LIMIT = 300;
const SESSION_RECALL_LEVEL_LIMIT = 5;
const SESSION_RECALL_SNIPPET_WIDTH = 200;
function extractSessionId(uri) {
    const match = uri.match(/^viking:\/\/session\/([^/]+)/);
    return match ? match[1] : null;
}
function snippetFromContent(content, pattern, width = SESSION_RECALL_SNIPPET_WIDTH) {
    const lower = content.toLowerCase();
    const idx = lower.indexOf(pattern.toLowerCase());
    if (idx < 0)
        return content.slice(0, width);
    const start = Math.max(0, idx - Math.floor(width / 3));
    const end = Math.min(content.length, start + width);
    const prefix = start > 0 ? "..." : "";
    const suffix = end < content.length ? "..." : "";
    return `${prefix}${content.slice(start, end).replace(/\s+/g, " ").trim()}${suffix}`;
}
server.tool("session_recall", "Find prior sessions whose transcript contains the given query. Substring (not semantic) match over session message bodies. Use when you need to recall an earlier conversation where a specific string, error, or topic was discussed.", {
    query: z.string().describe("Literal text to grep across session transcripts"),
    limit: z.number().optional().describe("Max sessions to return (default: 10)"),
    scope_uri: z.string().optional().describe("Session scope URI (default: viking://session)"),
    case_insensitive: z.boolean().optional().describe("Case-insensitive match (default: false)"),
}, async ({ query, limit, scope_uri, case_insensitive }) => {
    const cap = Math.max(1, Math.min(50, limit ?? 10));
    const scope = scope_uri ?? "viking://session";
    const { matches } = await client.grep(query, {
        uri: scope,
        nodeLimit: SESSION_RECALL_NODE_LIMIT,
        levelLimit: SESSION_RECALL_LEVEL_LIMIT,
        caseInsensitive: case_insensitive ?? false,
    });
    if (matches.length === 0) {
        return { content: [{ type: "text", text: `No sessions matched "${query}" under ${scope}.` }] };
    }
    const bySession = new Map();
    for (const m of matches) {
        const sid = extractSessionId(m.uri);
        if (!sid)
            continue;
        const entry = bySession.get(sid);
        if (entry) {
            entry.count += 1;
        }
        else {
            bySession.set(sid, {
                count: 1,
                firstSnippet: snippetFromContent(m.content, query),
                firstUri: m.uri,
            });
        }
    }
    if (bySession.size === 0) {
        return { content: [{ type: "text", text: `Matches found but none mapped to a session id under ${scope}.` }] };
    }
    const truncated = matches.length >= SESSION_RECALL_NODE_LIMIT;
    const top = [...bySession.entries()].sort((a, b) => b[1].count - a[1].count).slice(0, cap);
    const lines = top.map(([sid, info]) => `- ${sid} (${info.count} match${info.count === 1 ? "" : "es"})\n    ${info.firstSnippet}`);
    const note = truncated
        ? `\n\n(truncated at ${SESSION_RECALL_NODE_LIMIT} raw matches; refine query for full coverage)`
        : "";
    return {
        content: [{
                type: "text",
                text: `Found ${bySession.size} session${bySession.size === 1 ? "" : "s"} matching "${query}"${bySession.size > cap ? `, showing ${cap}` : ""}:\n\n${lines.join("\n")}${note}`,
            }],
    };
});
// -- Tool: memory_health --------------------------------------------------
server.tool("memory_health", "Check whether the OpenViking memory server is reachable and healthy.", {}, async () => {
    const ok = await client.healthCheck();
    return {
        content: [{
                type: "text",
                text: ok
                    ? `OpenViking is healthy (${config.baseUrl})`
                    : `OpenViking is unreachable at ${config.baseUrl}. Please check if the server is running.`,
            }],
    };
});
// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------
const transport = new StdioServerTransport();
await server.connect(transport);
