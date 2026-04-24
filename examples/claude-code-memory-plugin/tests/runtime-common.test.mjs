import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { access } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { computeSourceState, getRuntimePaths, syncRuntimeFiles } from "../scripts/runtime-common.mjs";

function withTempDir(fn) {
  const dir = mkdtempSync(join(tmpdir(), "openviking-cc-runtime-"));
  return Promise.resolve(fn(dir)).finally(() => {
    rmSync(dir, { recursive: true, force: true });
  });
}

function writeSourcePlugin(root) {
  mkdirSync(join(root, "servers"), { recursive: true });
  writeFileSync(join(root, "package.json"), JSON.stringify({ version: "0.1.5" }));
  writeFileSync(join(root, "package-lock.json"), JSON.stringify({ lockfileVersion: 3 }));
  writeFileSync(join(root, "servers", "memory-server.js"), 'import "./recall.js";\n');
  writeFileSync(join(root, "servers", "recall.js"), "export const ok = true;\n");
}

test("runtime sync copies all server modules required by memory-server", async () => {
  await withTempDir(async (dir) => {
    const pluginRoot = join(dir, "plugin");
    const pluginDataRoot = join(dir, "data");
    writeSourcePlugin(pluginRoot);

    const previousRoot = process.env.CLAUDE_PLUGIN_ROOT;
    const previousData = process.env.CLAUDE_PLUGIN_DATA;
    process.env.CLAUDE_PLUGIN_ROOT = pluginRoot;
    process.env.CLAUDE_PLUGIN_DATA = pluginDataRoot;

    try {
      const paths = getRuntimePaths();
      const state = await computeSourceState(paths);
      await syncRuntimeFiles(paths);

      await access(paths.runtimeServerPath);
      await access(paths.runtimeRecallPath);
      assert.match(state.serverHash, /^[a-f0-9]{64}$/);
    } finally {
      if (previousRoot === undefined) {
        delete process.env.CLAUDE_PLUGIN_ROOT;
      } else {
        process.env.CLAUDE_PLUGIN_ROOT = previousRoot;
      }

      if (previousData === undefined) {
        delete process.env.CLAUDE_PLUGIN_DATA;
      } else {
        process.env.CLAUDE_PLUGIN_DATA = previousData;
      }
    }
  });
});
