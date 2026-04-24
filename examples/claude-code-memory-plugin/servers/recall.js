const DEFAULT_MEMORY_SCOPES = ["viking://user/memories", "viking://agent/memories"];
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
function finalizeMemories(items, options) {
    return uniqueMemories(items)
        .filter(isRecallCandidate)
        .filter((item) => clampScore(item.score) >= options.scoreThreshold)
        .sort((left, right) => clampScore(right.score) - clampScore(left.score))
        .slice(0, options.limit);
}
export async function searchMemoryScopes(client, query, options) {
    if (options.targetUri) {
        const result = await client.find(query, {
            targetUri: options.targetUri,
            limit: options.limit,
            scoreThreshold: 0,
        });
        return {
            memories: finalizeMemories(result.memories ?? [], options),
            failedScopes: [],
        };
    }
    const settled = await Promise.allSettled(DEFAULT_MEMORY_SCOPES.map((scope) => client.find(query, { targetUri: scope, limit: options.limit, scoreThreshold: 0 })));
    const memories = [];
    const failedScopes = [];
    settled.forEach((result, index) => {
        const scope = DEFAULT_MEMORY_SCOPES[index];
        if (result.status === "fulfilled") {
            memories.push(...(result.value.memories ?? []));
        }
        else {
            failedScopes.push({ scope, reason: failureReason(result.reason) });
        }
    });
    if (memories.length === 0 && failedScopes.length === DEFAULT_MEMORY_SCOPES.length) {
        throw new Error(`OpenViking recall failed for all default memory scopes: ${formatScopeFailures(failedScopes)}`);
    }
    return {
        memories: finalizeMemories(memories, options),
        failedScopes,
    };
}
