import { describe, it } from "node:test"
import assert from "node:assert/strict"

import {
  buildRecallResponseText,
  isRecallCandidate,
  searchMemoryScopes,
} from "../servers/recall.js"

describe("Codex recall parity helpers", () => {
  it("keeps partial default-scope recall results when one scope fails", async () => {
    const client = {
      async find(_query, targetUri) {
        if (targetUri === "viking://user/memories") {
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
    const text = buildRecallResponseText("agent fact", result)

    assert.equal(result.memories.length, 1)
    assert.equal(result.failedScopes.length, 1)
    assert.match(text, /Agent fact/)
    assert.match(text, /Partial OpenViking memory results/)
  })

  it("filters non-leaf memories when level is present", async () => {
    assert.equal(isRecallCandidate({ uri: "viking://user/alice/memories", level: 0 }), false)
    assert.equal(isRecallCandidate({ uri: "viking://user/alice/memories/a.md", level: 2 }), true)
    assert.equal(isRecallCandidate({ uri: "viking://user/alice/memories/legacy.md" }), true)
  })

  it("adds a source note for order-like queries", () => {
    const text = buildRecallResponseText("what was the amazon order?", {
      memories: [],
      failedScopes: [],
    })

    assert.match(text, /order-like queries/)
    assert.match(text, /session_recall/)
  })
})
