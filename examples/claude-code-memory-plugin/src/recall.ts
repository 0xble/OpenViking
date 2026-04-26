export type FindResultItem = {
  uri: string;
  context_type?: "memory" | "resource" | "skill";
  level?: number;
  abstract?: string;
  overview?: string;
  category?: string;
  score?: number;
  match_reason?: string;
};

export type FindResult = {
  memories?: FindResultItem[];
  resources?: FindResultItem[];
  skills?: FindResultItem[];
  total?: number;
};

export type ScopeFailure = {
  scope: string;
  reason: string;
};

export type RecallSearchResult = {
  memories: FindResultItem[];
  resources: FindResultItem[];
  failedScopes: ScopeFailure[];
};

export type RecallClient = {
  find(query: string, options: { targetUri: string; limit: number; scoreThreshold?: number }): Promise<FindResult>;
};

const DEFAULT_MEMORY_SCOPES = ["viking://user/memories", "viking://agent/memories"] as const;
const DEFAULT_RESOURCE_SCOPE = "viking://resources";
const ORDER_QUERY_RE = /\b(order|ordered|purchase|purchased|bought|buy|item|product|amazon|shop|shipping|delivery)\b/i;

export function clampScore(value: number | undefined): number {
  if (typeof value !== "number" || Number.isNaN(value)) return 0;
  return Math.max(0, Math.min(1, value));
}

export function isRecallCandidate(item: FindResultItem): boolean {
  return item.level === undefined || item.level === 2;
}

export function isOrderLikeQuery(query: string): boolean {
  return ORDER_QUERY_RE.test(query);
}

export function formatScopeFailures(failures: ScopeFailure[]): string {
  if (failures.length === 0) return "";
  const scopes = failures.map((failure) => `${failure.scope}: ${failure.reason}`).join("; ");
  return `Partial OpenViking memory results. Some scopes failed: ${scopes}.`;
}

export function formatOrderSourceNote(query: string): string {
  if (!isOrderLikeQuery(query)) return "";
  return "Source note: order-like queries may need session_recall or resources if the item was never stored as memory.";
}

function failureReason(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function uniqueMemories(items: FindResultItem[]): FindResultItem[] {
  return uniqueItems(items);
}

function uniqueItems(items: FindResultItem[]): FindResultItem[] {
  const seen = new Set<string>();
  const result: FindResultItem[] = [];
  for (const item of items) {
    if (seen.has(item.uri)) continue;
    seen.add(item.uri);
    result.push(item);
  }
  return result;
}

function withContextType(items: FindResultItem[], contextType: "memory" | "resource" | "skill"): FindResultItem[] {
  return items.map((item) => ({ ...item, context_type: item.context_type ?? contextType }));
}

function finalizeMemories(
  items: FindResultItem[],
  options: { limit: number; scoreThreshold: number },
): FindResultItem[] {
  return uniqueMemories(items)
    .filter(isRecallCandidate)
    .filter((item) => clampScore(item.score) >= options.scoreThreshold)
    .sort((left, right) => clampScore(right.score) - clampScore(left.score))
    .slice(0, options.limit);
}

function finalizeResources(
  items: FindResultItem[],
  options: { limit: number; scoreThreshold: number },
): FindResultItem[] {
  return uniqueItems(items)
    .filter(isRecallCandidate)
    .filter((item) => clampScore(item.score) >= options.scoreThreshold)
    .sort((left, right) => clampScore(right.score) - clampScore(left.score))
    .slice(0, options.limit);
}

function isResourceScope(uri: string | undefined): boolean {
  const normalized = uri?.trim().replace(/\/+$/, "") ?? "";
  return normalized === DEFAULT_RESOURCE_SCOPE || normalized.startsWith(`${DEFAULT_RESOURCE_SCOPE}/`);
}

export async function searchMemoryScopes(
  client: RecallClient,
  query: string,
  options: { targetUri?: string; limit: number; scoreThreshold: number; includeResources?: boolean },
): Promise<RecallSearchResult> {
  if (options.targetUri) {
    const result = await client.find(query, {
      targetUri: options.targetUri,
      limit: options.limit,
      scoreThreshold: 0,
    });
    return {
      memories: finalizeMemories(withContextType(result.memories ?? [], "memory"), options),
      resources: isResourceScope(options.targetUri)
        ? finalizeResources(withContextType(result.resources ?? [], "resource"), options)
        : [],
      failedScopes: [],
    };
  }

  const scopes = options.includeResources
    ? [...DEFAULT_MEMORY_SCOPES, DEFAULT_RESOURCE_SCOPE]
    : [...DEFAULT_MEMORY_SCOPES];
  const settled = await Promise.allSettled(
    scopes.map((scope) =>
      client.find(query, { targetUri: scope, limit: options.limit, scoreThreshold: 0 }),
    ),
  );
  const memories: FindResultItem[] = [];
  const resources: FindResultItem[] = [];
  const failedScopes: ScopeFailure[] = [];

  settled.forEach((result, index) => {
    const scope = scopes[index]!;
    if (result.status === "fulfilled") {
      memories.push(...withContextType(result.value.memories ?? [], "memory"));
      if (isResourceScope(scope)) {
        resources.push(...withContextType(result.value.resources ?? [], "resource"));
      }
    } else {
      failedScopes.push({ scope, reason: failureReason(result.reason) });
    }
  });

  if (memories.length === 0 && resources.length === 0 && failedScopes.length === scopes.length) {
    throw new Error(`OpenViking recall failed for all default memory scopes: ${formatScopeFailures(failedScopes)}`);
  }

  return {
    memories: finalizeMemories(memories, options),
    resources: finalizeResources(resources, options),
    failedScopes,
  };
}
