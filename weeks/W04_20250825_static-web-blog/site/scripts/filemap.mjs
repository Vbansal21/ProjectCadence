import { promises as fs } from "node:fs";
import { join, relative } from "node:path";

const root = "public";
const rows = [];

async function walk(dir) {
  const entries = await fs.readdir(dir, { withFileTypes: true });
  for (const e of entries) {
    const p = join(dir, e.name);
    if (e.isDirectory()) await walk(p);
    else rows.push(p);
  }
}
await walk(root);
rows.sort();

const items = rows.map(p => {
  const href = relative(root, p).replace(/\\/g, "/");
  return `<li><a href="./${href}">${href}</a></li>`;
}).join("\n");

const html = `<!doctype html>
<meta charset="utf-8">
<title>__files__</title>
<style>body{font:14px ui-sans-serif,system-ui;background:#0a0a0a;color:#e5e5e5;padding:24px}
a{color:#93c5fd;text-decoration:none}a:hover{text-decoration:underline}
code{background:#111827;padding:.1rem .25rem;border-radius:.25rem}
</style>
<h1>Published files</h1>
<p>Base path is this page's directory. If a link below 404s, the file wasn't deployed.</p>
<ul>${items}</ul>`;
await fs.writeFile(join(root, "__files__.html"), html);
console.log(`Wrote ${rows.length} entries to public/__files__.html`);
