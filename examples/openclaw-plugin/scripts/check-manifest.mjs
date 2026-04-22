#!/usr/bin/env node
// Verify every relative "./X.js" import in the plugin's runtime .ts files
// is listed in install-manifest.json (required or optional). Catches the
// silent-install-break class of bug where a new source file is added but
// the installer manifest is not updated, so fresh installs download a
// subset that fails to load at runtime.

import { readFile, readdir, stat } from "node:fs/promises";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const pluginDir = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const manifestPath = join(pluginDir, "install-manifest.json");

const ignoredSources = new Set([
  "vitest.config.ts",
]);

const ignoredDirs = new Set([
  "node_modules",
  "scripts",
  "tests",
  "__tests__",
  "skills",
  "setup-helper",
  "upgrade_scripts",
  "health_check_tools",
  "images",
]);

const importPattern = /from\s+["'](\.\/[^"']+\.js)["']/g;

const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
const manifestFiles = new Set([
  ...(manifest.files?.required ?? []),
  ...(manifest.files?.optional ?? []),
]);

async function collectSourceFiles(dir) {
  const out = [];
  const entries = await readdir(dir, { withFileTypes: true });
  for (const entry of entries) {
    if (entry.name.startsWith(".")) continue;
    if (entry.isDirectory()) {
      if (ignoredDirs.has(entry.name)) continue;
      const nested = await collectSourceFiles(join(dir, entry.name));
      out.push(...nested);
      continue;
    }
    if (!entry.isFile()) continue;
    if (!entry.name.endsWith(".ts")) continue;
    if (ignoredSources.has(entry.name)) continue;
    out.push(join(dir, entry.name));
  }
  return out;
}

const sourceFiles = (await collectSourceFiles(pluginDir)).map((abs) =>
  relative(pluginDir, abs),
);

const missing = [];
const referenced = new Set();

for (const file of sourceFiles) {
  const src = await readFile(join(pluginDir, file), "utf8");
  const importerDir = dirname(file);
  for (const match of src.matchAll(importPattern)) {
    const jsPath = match[1].slice(2);
    const relTs = jsPath.replace(/\.js$/, ".ts");
    const tsPath = importerDir === "."
      ? relTs
      : relative(pluginDir, resolve(pluginDir, importerDir, relTs));
    referenced.add(tsPath);
    if (!manifestFiles.has(tsPath)) {
      missing.push({ importer: file, imports: jsPath, expected: tsPath });
    }
  }
}

// Optionally also enforce that every .ts file the manifest advertises
// actually exists on disk.
const missingOnDisk = [];
for (const entry of manifestFiles) {
  if (!entry.endsWith(".ts")) continue;
  const exists = sourceFiles.includes(entry);
  if (!exists) missingOnDisk.push(entry);
}

let failed = false;

if (missing.length) {
  failed = true;
  console.error("install-manifest.json is missing files imported by runtime sources:");
  for (const m of missing) {
    console.error(`  ${m.importer} imports ${m.imports} -> add "${m.expected}" to files.optional`);
  }
}

if (missingOnDisk.length) {
  failed = true;
  console.error("install-manifest.json lists files that no longer exist on disk:");
  for (const entry of missingOnDisk) console.error(`  ${entry}`);
}

if (failed) {
  process.exit(1);
}

console.error(
  `install-manifest.json OK (${sourceFiles.length} ts sources, ${referenced.size} relative imports)`,
);
