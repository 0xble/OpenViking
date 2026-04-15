import { describe, expect, it, vi } from "vitest";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import type { OpenVikingClient } from "../../client.js";
import { memoryOpenVikingConfigSchema } from "../../config.js";
import { createMemoryOpenVikingContextEngine } from "../../context-engine.js";

function makeLogger() {
  return {
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  };
}

function makeEngine(commitResult: unknown, opts?: { throwError?: Error }) {
  const cfg = memoryOpenVikingConfigSchema.parse({
    mode: "remote",
    baseUrl: "http://127.0.0.1:1933",
    autoCapture: false,
    autoRecall: false,
    ingestReplyAssist: false,
  });
  const logger = makeLogger();

  const commitSession = opts?.throwError
    ? vi.fn().mockRejectedValue(opts.throwError)
    : vi.fn().mockResolvedValue(commitResult);

  const client = {
    commitSession,
    getSessionContext: vi.fn().mockResolvedValue({
      latest_archive_overview: "",
      latest_archive_id: "",
      pre_archive_abstracts: [],
      messages: [],
      estimatedTokens: 0,
      stats: { totalArchives: 0, includedArchives: 0, droppedArchives: 0, failedArchives: 0, activeTokens: 0, archiveTokens: 0 },
    }),
  } as unknown as OpenVikingClient;

  const getClient = vi.fn().mockResolvedValue(client);
  const resolveAgentId = vi.fn((_sid: string) => "test-agent");

  const engine = createMemoryOpenVikingContextEngine({
    id: "openviking",
    name: "Test Engine",
    version: "test",
    cfg,
    logger,
    getClient,
    resolveAgentId,
  });

  return {
    engine,
    client: client as unknown as {
      commitSession: ReturnType<typeof vi.fn>;
    },
    logger,
  };
}

describe("context-engine commitOVSession()", () => {
  it("returns true on successful commit", async () => {
    const { engine } = makeEngine({
      status: "completed",
      archived: false,
      memories_extracted: { core: 1 },
    });

    const ok = await engine.commitOVSession("test-session");
    expect(ok).toBe(true);
  });

  it("returns false on failed commit", async () => {
    const { engine } = makeEngine({
      status: "failed",
      error: "extraction error",
    });

    const ok = await engine.commitOVSession("test-session");
    expect(ok).toBe(false);
  });

  it("returns false on timeout commit", async () => {
    const { engine } = makeEngine({
      status: "timeout",
      task_id: "task-timeout",
    });

    const ok = await engine.commitOVSession("test-session");
    expect(ok).toBe(false);
  });

  it("returns false when commit throws", async () => {
    const { engine } = makeEngine(null, {
      throwError: new Error("connection refused"),
    });

    const ok = await engine.commitOVSession("test-session");
    expect(ok).toBe(false);
  });

  it("uses wait=true for synchronous extraction", async () => {
    const { engine, client } = makeEngine({
      status: "completed",
      archived: false,
      memories_extracted: {},
    });

    await engine.commitOVSession("s1");

    expect(client.commitSession.mock.calls[0][1]).toMatchObject({ wait: true });
  });

  it("logs memories extracted count", async () => {
    const { engine, logger } = makeEngine({
      status: "completed",
      archived: true,
      memories_extracted: { core: 3, preferences: 1 },
    });

    await engine.commitOVSession("s1");

    expect(logger.info).toHaveBeenCalledWith(
      expect.stringContaining("memories=4"),
    );
  });

  it("skips commitOVSession when the session matches bypassSessionPatterns", async () => {
    const cfg = memoryOpenVikingConfigSchema.parse({
      mode: "remote",
      baseUrl: "http://127.0.0.1:1933",
      autoCapture: false,
      autoRecall: false,
      ingestReplyAssist: false,
      bypassSessionPatterns: ["agent:*:cron:**"],
    });
    const logger = makeLogger();
    const getClient = vi.fn();
    const resolveAgentId = vi.fn((_sid: string) => "test-agent");

    const engine = createMemoryOpenVikingContextEngine({
      id: "openviking",
      name: "Test Engine",
      version: "test",
      cfg,
      logger,
      getClient: getClient as any,
      resolveAgentId,
    });

    const ok = await engine.commitOVSession("runtime-session", "agent:main:cron:nightly:run:1");

    expect(ok).toBe(false);
    expect(getClient).not.toHaveBeenCalled();
    expect(logger.warn).toHaveBeenCalledWith(
      expect.stringContaining("session is bypassed"),
    );
  });
});

describe("context-engine compact()", () => {
  it("returns compacted=false when the session matches bypassSessionPatterns", async () => {
    const cfg = memoryOpenVikingConfigSchema.parse({
      mode: "remote",
      baseUrl: "http://127.0.0.1:1933",
      autoCapture: false,
      autoRecall: false,
      ingestReplyAssist: false,
      bypassSessionPatterns: ["agent:*:cron:**"],
    });
    const logger = makeLogger();
    const getClient = vi.fn();
    const resolveAgentId = vi.fn((_sid: string) => "test-agent");

    const engine = createMemoryOpenVikingContextEngine({
      id: "openviking",
      name: "Test Engine",
      version: "test",
      cfg,
      logger,
      getClient: getClient as any,
      resolveAgentId,
    });

    const result = await engine.compact({
      sessionId: "agent:main:cron:nightly:run:1",
      sessionFile: "",
    });

    expect(result).toEqual({
      ok: true,
      compacted: false,
      reason: "session_bypassed",
    });
    expect(getClient).not.toHaveBeenCalled();
  });

  it("returns compacted=true when commit succeeds with archived=true", async () => {
    const { engine } = makeEngine({
      status: "completed",
      archived: true,
      task_id: "task-1",
      memories_extracted: { core: 3, preferences: 1 },
    });

    const result = await engine.compact({
      sessionId: "s1",
      sessionFile: "",
    });

    expect(result.ok).toBe(true);
    expect(result.compacted).toBe(true);
    expect(result.reason).toBe("commit_completed");
  });

  it("returns compacted=false when commit succeeds with archived=false", async () => {
    const { engine } = makeEngine({
      status: "completed",
      archived: false,
      task_id: "task-2",
      memories_extracted: {},
    });

    const result = await engine.compact({
      sessionId: "s2",
      sessionFile: "",
    });

    expect(result.ok).toBe(true);
    expect(result.compacted).toBe(false);
    expect(result.reason).toBe("commit_no_archive");
  });

  it("uses local fallback when commit status is 'failed'", async () => {
    const { engine, logger } = makeEngine({
      status: "failed",
      error: "extraction pipeline error",
      task_id: "task-3",
    });

    const result = await engine.compact({
      sessionId: "s3",
      sessionFile: "",
    });

    expect(result.ok).toBe(true);
    expect(result.compacted).toBe(true);
    expect(result.reason).toBe("local_fallback_after_commit_failed");
    expect(result.result?.summary).toContain("Local fallback compaction summary");
    expect(logger.warn).toHaveBeenCalledWith(
      expect.stringContaining("Phase 2 failed"),
    );
  });

  it("uses local fallback when commit status is 'timeout'", async () => {
    const { engine, logger } = makeEngine({
      status: "timeout",
      task_id: "task-4",
    });

    const result = await engine.compact({
      sessionId: "s4",
      sessionFile: "",
    });

    expect(result.ok).toBe(true);
    expect(result.compacted).toBe(true);
    expect(result.reason).toBe("local_fallback_after_commit_timeout");
    expect(result.result?.firstKeptEntryId).toBe("local-fallback");
    expect(logger.warn).toHaveBeenCalledWith(
      expect.stringContaining("Phase 2 timed out"),
    );
  });

  it("local fallback strips large transcript payloads", async () => {
    const { engine } = makeEngine({
      status: "timeout",
      task_id: "task-large",
    });
    const dir = await mkdtemp(join(tmpdir(), "openviking-plugin-"));
    const sessionFile = join(dir, "session.jsonl");
    const largeBlob = "A".repeat(20_000);
    const transcript = [
      {
        role: "user",
        content: [
          {
            type: "input_text",
            text: `Please inspect this image data:image/png;base64,${largeBlob}`,
          },
          { type: "input_image", image_url: `data:image/png;base64,${largeBlob}` },
        ],
      },
    ].map((entry) => JSON.stringify(entry)).join("\n");

    try {
      await writeFile(sessionFile, transcript);

      const result = await engine.compact({
        sessionId: "s-large",
        sessionFile,
      });

      expect(result.ok).toBe(true);
      expect(result.compacted).toBe(true);
      expect(result.reason).toBe("local_fallback_after_commit_timeout");
      expect(result.result?.summary).toContain("[large data URL omitted]");
      expect(result.result?.summary).not.toContain(largeBlob);
      expect((result.result?.summary ?? "").length).toBeLessThanOrEqual(16_000);
      expect(result.result?.tokensAfter).toBeLessThan(10_000);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  it("commit passes wait=true for synchronous extraction", async () => {
    const { engine, client } = makeEngine({
      status: "completed",
      archived: true,
      memories_extracted: { core: 2 },
    });

    await engine.compact({ sessionId: "s1", sessionFile: "" });

    expect(client.commitSession).toHaveBeenCalledTimes(1);
    expect(client.commitSession.mock.calls[0][1]).toMatchObject({ wait: true });
  });

  it("logs memory extraction count on success", async () => {
    const { engine, logger } = makeEngine({
      status: "completed",
      archived: true,
      task_id: "task-mem",
      memories_extracted: { core: 5, preferences: 2 },
    });

    await engine.compact({ sessionId: "s1", sessionFile: "" });

    expect(logger.info).toHaveBeenCalledWith(
      expect.stringContaining("memories=7"),
    );
  });

  it("handles commit with zero memories extracted", async () => {
    const { engine, logger } = makeEngine({
      status: "completed",
      archived: true,
      task_id: "task-empty",
      memories_extracted: {},
    });

    await engine.compact({ sessionId: "s-empty", sessionFile: "" });

    expect(logger.info).toHaveBeenCalledWith(
      expect.stringContaining("memories=0"),
    );
  });

  it("handles commit with missing memories_extracted field", async () => {
    const { engine } = makeEngine({
      status: "completed",
      archived: false,
    });

    const result = await engine.compact({ sessionId: "s-no-mem", sessionFile: "" });
    expect(result.ok).toBe(true);
  });

  it("uses correct OV session ID derived from sessionId", async () => {
    const { engine, client } = makeEngine({
      status: "completed",
      archived: false,
      memories_extracted: {},
    });

    await engine.compact({
      sessionId: "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      sessionFile: "",
    });

    const commitCallSessionId = client.commitSession.mock.calls[0][0] as string;
    expect(commitCallSessionId).toBe("a1b2c3d4-e5f6-7890-abcd-ef1234567890");
  });

  it("passes agentId to commitSession", async () => {
    const { engine, client } = makeEngine({
      status: "completed",
      archived: false,
      memories_extracted: {},
    });

    await engine.compact({ sessionId: "s1", sessionFile: "" });

    expect(client.commitSession.mock.calls[0][1]).toMatchObject({
      agentId: "test-agent",
    });
  });

  it("uses local fallback when commit throws", async () => {
    const { engine, logger } = makeEngine(null, {
      throwError: new Error("network unreachable"),
    });

    const result = await engine.compact({
      sessionId: "s5",
      sessionFile: "",
    });

    expect(result.ok).toBe(true);
    expect(result.compacted).toBe(true);
    expect(result.reason).toBe("local_fallback_after_commit_error");
    expect(logger.warn).toHaveBeenCalledWith(
      expect.stringContaining("commit failed"),
    );
  });
});
