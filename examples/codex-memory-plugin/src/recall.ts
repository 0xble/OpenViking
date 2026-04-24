export type FindResultItem = {
  uri: string
  level?: number
  abstract?: string
  overview?: string
  category?: string
  score?: number
}

export type FindResult = {
  memories?: FindResultItem[]
  resources?: FindResultItem[]
  skills?: FindResultItem[]
}

export type ScopeFailure = {
  scope: string
  reason: string
}

export type RecallSearchResult = {
  memories: FindResultItem[]
  failedScopes: ScopeFailure[]
}

export type RecallClient = {
  find(query: string, targetUri: string, limit: number, scoreThreshold: number): Promise<FindResult>
}

const DEFAULT_MEMORY_SCOPES = ["viking://user/memories", "viking://agent/memories"] as const
const ORDER_QUERY_RE = /\b(order|ordered|purchase|purchased|bought|buy|item|product|amazon|shop|shipping|delivery)\b/i

export function clampScore(value: number | undefined): number {
  if (typeof value !== "number" || Number.isNaN(value)) return 0
  return Math.max(0, Math.min(1, value))
}

export function isRecallCandidate(item: FindResultItem): boolean {
  return item.level === undefined || item.level === 2
}

export function isOrderLikeQuery(query: string): boolean {
  return ORDER_QUERY_RE.test(query)
}

export function formatMemoryResults(items: FindResultItem[]): string {
  return items
    .map((item, index) => {
      const summary = item.abstract?.trim() || item.overview?.trim() || item.uri
      const score = Math.round(clampScore(item.score) * 100)
      return `${index + 1}. ${summary}\n   URI: ${item.uri}\n   Score: ${score}%`
    })
    .join("\n\n")
}

export function formatScopeFailures(failures: ScopeFailure[]): string {
  if (failures.length === 0) return ""
  const scopes = failures.map((failure) => `${failure.scope}: ${failure.reason}`).join("; ")
  return `Partial OpenViking memory results. Some scopes failed: ${scopes}.`
}

function failureReason(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function uniqueMemories(items: FindResultItem[]): FindResultItem[] {
  const seen = new Set<string>()
  const result: FindResultItem[] = []
  for (const item of items) {
    if (seen.has(item.uri)) continue
    seen.add(item.uri)
    result.push(item)
  }
  return result
}

function finalizeMemories(
  items: FindResultItem[],
  options: { limit: number; scoreThreshold: number },
): FindResultItem[] {
  return uniqueMemories(items)
    .filter(isRecallCandidate)
    .filter((item) => clampScore(item.score) >= options.scoreThreshold)
    .sort((left, right) => clampScore(right.score) - clampScore(left.score))
    .slice(0, options.limit)
}

export async function searchMemoryScopes(
  client: RecallClient,
  query: string,
  options: { targetUri?: string; limit: number; scoreThreshold: number },
): Promise<RecallSearchResult> {
  if (options.targetUri) {
    const result = await client.find(query, options.targetUri, options.limit, 0)
    return {
      memories: finalizeMemories(result.memories ?? [], options),
      failedScopes: [],
    }
  }

  const settled = await Promise.allSettled(
    DEFAULT_MEMORY_SCOPES.map((scope) => client.find(query, scope, options.limit, 0)),
  )
  const memories: FindResultItem[] = []
  const failedScopes: ScopeFailure[] = []

  settled.forEach((result, index) => {
    const scope = DEFAULT_MEMORY_SCOPES[index]!
    if (result.status === "fulfilled") {
      memories.push(...(result.value.memories ?? []))
    } else {
      failedScopes.push({ scope, reason: failureReason(result.reason) })
    }
  })

  if (memories.length === 0 && failedScopes.length === DEFAULT_MEMORY_SCOPES.length) {
    throw new Error(`OpenViking recall failed for all default memory scopes: ${formatScopeFailures(failedScopes)}`)
  }

  return {
    memories: finalizeMemories(memories, options),
    failedScopes,
  }
}

export function buildRecallResponseText(query: string, result: RecallSearchResult): string {
  const notes = [
    formatScopeFailures(result.failedScopes),
    isOrderLikeQuery(query)
      ? "Source note: order-like queries may need session_recall or resources if the item was never stored as memory."
      : "",
  ].filter(Boolean)

  if (result.memories.length === 0) {
    return ["No relevant OpenViking memories found.", ...notes].join("\n\n")
  }

  return [formatMemoryResults(result.memories), ...notes].join("\n\n")
}
