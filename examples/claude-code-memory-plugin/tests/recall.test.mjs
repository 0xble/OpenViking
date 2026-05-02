import { describe, it } from "node:test"
import assert from "node:assert/strict"

import {
  formatOrderSourceNote,
  formatScopeFailures,
  isRecallCandidate,
  searchMemoryScopes,
  searchResourceScope,
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
    assert.equal(result.resources.length, 0)
    assert.equal(result.failedScopes.length, 1)
    assert.match(formatScopeFailures(result.failedScopes), /Partial OpenViking memory results/)
  })

  it("returns resources for an explicit resource target", async () => {
    const client = {
      async find(_query, options) {
        assert.equal(options.targetUri, "viking://resources/docs")
        return {
          resources: [
            { uri: "viking://resources/docs/api.md", level: 2, abstract: "API notes", score: 0.8 },
          ],
        }
      },
    }

    const result = await searchMemoryScopes(client, "api notes", {
      targetUri: "viking://resources/docs",
      limit: 6,
      scoreThreshold: 0.01,
    })

    assert.equal(result.memories.length, 0)
    assert.equal(result.resources.length, 1)
    assert.equal(result.resources[0].context_type, "resource")
  })

  it("searches viking resources by default for resource recall", async () => {
    const client = {
      async find(_query, options) {
        assert.equal(options.targetUri, "viking://resources")
        return {
          resources: [
            { uri: "viking://resources/slack/messages.json", level: 2, abstract: "Slack evidence", score: 0.9 },
          ],
        }
      },
    }

    const result = await searchResourceScope(client, "slack evidence", {
      limit: 6,
      scoreThreshold: 0.01,
    })

    assert.equal(result.memories.length, 0)
    assert.equal(result.resources.length, 1)
    assert.equal(result.resources[0].context_type, "resource")
  })

  it("filters weak resource recall matches", async () => {
    const client = {
      async find(_query, options) {
        assert.equal(options.targetUri, "viking://resources")
        return {
          resources: [
            { uri: "viking://resources/email/unrelated.json", level: 2, abstract: "Unrelated email", score: 0.34 },
            { uri: "viking://resources/docs/relevant.md", level: 2, abstract: "Relevant project notes", score: 0.72 },
          ],
        }
      },
    }

    const result = await searchResourceScope(client, "project notes", {
      limit: 6,
      scoreThreshold: 0.4,
    })

    assert.equal(result.resources.length, 1)
    assert.equal(result.resources[0].uri, "viking://resources/docs/relevant.md")
    assert.equal(result.resources[0].abstract, "Relevant project notes")
  })

  it("does not treat resource-like prefixes as resource scopes", async () => {
    const client = {
      async find(_query, options) {
        assert.equal(options.targetUri, "viking://resourcesFoo/docs")
        return {
          resources: [
            { uri: "viking://resourcesFoo/docs/api.md", level: 2, abstract: "API notes", score: 0.8 },
          ],
        }
      },
    }

    const result = await searchMemoryScopes(client, "api notes", {
      targetUri: "viking://resourcesFoo/docs",
      limit: 6,
      scoreThreshold: 0.01,
    })

    assert.equal(result.resources.length, 0)
  })

  it("keeps default unscoped recall memory-only unless resources are requested", async () => {
    const seen = []
    const client = {
      async find(_query, options) {
        seen.push(options.targetUri)
        return { memories: [], resources: [] }
      },
    }

    await searchMemoryScopes(client, "api notes", { limit: 6, scoreThreshold: 0.01 })
    assert.deepEqual(seen, ["viking://user/memories", "viking://agent/memories"])
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
