export type FindResultItem = {
  uri: string;
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
  failedScopes: ScopeFailure[];
};

export type RecallClient = {
  find(query: string, options: { targetUri: string; limit: number; scoreThreshold?: number }): Promise<FindResult>;
};

const DEFAULT_MEMORY_SCOPES = ["viking://user/memories", "viking://agent/memories"] as const;
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
  const seen = new Set<string>();
  const result: FindResultItem[] = [];
  for (const item of items) {
    if (seen.has(item.uri)) continue;
    seen.add(item.uri);
    result.push(item);
  }
  return result;
}

export async function searchMemoryScopes(
  client: RecallClient,
  query: string,
  options: { targetUri?: string; limit: number; scoreThreshold: number },
): Promise<RecallSearchResult> {
  if (options.targetUri) {
    const result = await client.find(query, {
      targetUri: options.targetUri,
      limit: options.limit,
      scoreThreshold: 0,
    });
    return {
      memories: uniqueMemories(result.memories ?? []).filter(isRecallCandidate),
      failedScopes: [],
    };
  }

  const settled = await Promise.allSettled(
    DEFAULT_MEMORY_SCOPES.map((scope) =>
      client.find(query, { targetUri: scope, limit: options.limit, scoreThreshold: 0 }),
    ),
  );
  const memories: FindResultItem[] = [];
  const failedScopes: ScopeFailure[] = [];

  settled.forEach((result, index) => {
    const scope = DEFAULT_MEMORY_SCOPES[index]!;
    if (result.status === "fulfilled") {
      memories.push(...(result.value.memories ?? []));
    } else {
      failedScopes.push({ scope, reason: failureReason(result.reason) });
    }
  });

  if (memories.length === 0 && failedScopes.length === DEFAULT_MEMORY_SCOPES.length) {
    throw new Error(`OpenViking recall failed for all default memory scopes: ${formatScopeFailures(failedScopes)}`);
  }

  return {
    memories: uniqueMemories(memories).filter(isRecallCandidate),
    failedScopes,
  };
}
