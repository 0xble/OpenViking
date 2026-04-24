import { describe, it } from "node:test"
import assert from "node:assert/strict"

import {
  formatOrderSourceNote,
  formatScopeFailures,
  isRecallCandidate,
  searchMemoryScopes,
} from "../servers/recall.js"

describe("Claude Code MCP recall parity helpers", () => {
  it("keeps partial default-scope recall results when one scope fails", async () => {
    const client = {
      async find(_query, options) {
        if (options.targetUri === "viking://user/memories") {
          throw new Error("user scope timeout")
        }
        return {
          memories: [
            { uri: "viking://agent/worker/memories/a.md", level: 2, abstract: "Agent fact", score: 0.7 },
          ],
        }
      },
    }

    const result = await searchMemoryScopes(client, "agent fact", {
      limit: 6,
      scoreThreshold: 0.01,
    })

    assert.equal(result.memories.length, 1)
    assert.equal(result.failedScopes.length, 1)
    assert.match(formatScopeFailures(result.failedScopes), /Partial OpenViking memory results/)
  })

  it("filters non-leaf memories when level is present", async () => {
    assert.equal(isRecallCandidate({ uri: "viking://user/alice/memories", level: 0 }), false)
    assert.equal(isRecallCandidate({ uri: "viking://user/alice/memories/a.md", level: 2 }), true)
    assert.equal(isRecallCandidate({ uri: "viking://user/alice/memories/legacy.md" }), true)
  })

  it("adds a source note for order-like queries", () => {
    assert.match(formatOrderSourceNote("what did I order?"), /order-like queries/)
    assert.equal(formatOrderSourceNote("memory preferences"), "")
  })
})
