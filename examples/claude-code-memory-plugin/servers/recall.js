const DEFAULT_MEMORY_SCOPES = ["viking://user/memories", "viking://agent/memories"];
const DEFAULT_RESOURCE_SCOPE = "viking://resources";
const ORDER_QUERY_RE = /\b(order|ordered|purchase|purchased|bought|buy|item|product|amazon|shop|shipping|delivery)\b/i;
export function clampScore(value) {
    if (typeof value !== "number" || Number.isNaN(value))
        return 0;
    return Math.max(0, Math.min(1, value));
}
export function isRecallCandidate(item) {
    return item.level === undefined || item.level === 2;
}
export function isOrderLikeQuery(query) {
    return ORDER_QUERY_RE.test(query);
}
export function formatScopeFailures(failures) {
    if (failures.length === 0)
        return "";
    const scopes = failures.map((failure) => `${failure.scope}: ${failure.reason}`).join("; ");
    return `Partial OpenViking memory results. Some scopes failed: ${scopes}.`;
}
export function formatOrderSourceNote(query) {
    if (!isOrderLikeQuery(query))
        return "";
    return "Source note: order-like queries may need session_recall or resources if the item was never stored as memory.";
}
function failureReason(error) {
    return error instanceof Error ? error.message : String(error);
}
function uniqueMemories(items) {
    return uniqueItems(items);
}
function uniqueItems(items) {
    const seen = new Set();
    const result = [];
    for (const item of items) {
        if (seen.has(item.uri))
            continue;
        seen.add(item.uri);
        result.push(item);
    }
    return result;
}
function withContextType(items, contextType) {
    return items.map((item) => ({ ...item, context_type: item.context_type ?? contextType }));
}
function finalizeMemories(items, options) {
    return uniqueMemories(items)
        .filter(isRecallCandidate)
        .filter((item) => clampScore(item.score) >= options.scoreThreshold)
        .sort((left, right) => clampScore(right.score) - clampScore(left.score))
        .slice(0, options.limit);
}
function finalizeResources(items, options) {
    return uniqueItems(items)
        .filter(isRecallCandidate)
        .filter((item) => clampScore(item.score) >= options.scoreThreshold)
        .sort((left, right) => clampScore(right.score) - clampScore(left.score))
        .slice(0, options.limit);
}
function isResourceScope(uri) {
    const normalized = uri?.trim().replace(/\/+$/, "") ?? "";
    return normalized === DEFAULT_RESOURCE_SCOPE || normalized.startsWith(`${DEFAULT_RESOURCE_SCOPE}/`);
}
export async function searchMemoryScopes(client, query, options) {
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
    const settled = await Promise.allSettled(scopes.map((scope) => client.find(query, { targetUri: scope, limit: options.limit, scoreThreshold: 0 })));
    const memories = [];
    const resources = [];
    const failedScopes = [];
    settled.forEach((result, index) => {
        const scope = scopes[index];
        if (result.status === "fulfilled") {
            memories.push(...withContextType(result.value.memories ?? [], "memory"));
            if (isResourceScope(scope)) {
                resources.push(...withContextType(result.value.resources ?? [], "resource"));
            }
        }
        else {
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
