# Week 4 — Dev Notes / Log

**Theme:** Re-center on fundamentals and still look modern. Ship a static blog that proves I understand semantics, responsive layout, and progressive enhancement—with small, defensible bits of JS and D3 where it earns its keep.

**Dates:** 2025-08-25 → 2025-08-30

---

## Day-by-day

### Mon (Aug 25) — Framing & scaffold
- Set the weekly folder: `weeks/W04_20250825_web-blog/`.
- Decided on **project page** deployment (no Flask; static only).
- Chose the tech: **HTMX** for partials, **Tailwind** for styling discipline, **TypeScript** for ergonomics, **D3** for tiny explainers.
- Scaffolded `site/` with `public/` (served) and `src/` (inputs). Added stub assets to avoid 404s pre-build.

**Learned/affirmed**
- Relative paths everywhere make project-page hosting painless.
- Tiny, page-local scripts beat a global bundle for this style of site.

### Tue (Aug 26) — Build pipeline & Pages
- Wrote `package.json`, Tailwind config, TS config; `npm install` to create the lockfile.
- Added `.github/workflows/deployW04.yml` that builds from the subdir and publishes `public/` to `gh-pages`.
- Turned on “Read and write” workflow permissions; Pages → `gh-pages`.

**Snags**
- `npm ci` without a lockfile fails (expected). Fixed by running `npm install` once and committing `package-lock.json`.
- Reminder: Pages won’t compile anything—CI must output ready-to-serve static files.

### Wed (Aug 27) — Post 1 (Semantics) + live audit
- Wrote **Semantic HTML for A11y & SEO**.
- Implemented a **self-audit** that computes landmarks/headings/alt/forms/table/meta from the article’s own DOM and renders it with **D3**.
- Kept SVG colors inline (`fill`, `stroke`) to avoid Tailwind purge surprises.

**What improved my mental model**
- ARIA is a supplement, not a patch for ignoring native elements.
- A tiny, honest visualization can double as documentation and a guardrail.

### Thu (Aug 28) — Post 2 (Responsive) + readability probe
- Wrote **Responsive Design: Fluid Type & Grid**.
- Added **CPL probe** (characters per line) measured against a target band, updating on resize via **D3**.
- Documented `clamp()`, `minmax()`, `auto-fit/auto-fill`, container queries, and `dvh`.

**Notes to self**
- Container queries flip a lot of old “viewport-only” thinking.
- CPL as a live metric nudges real typography choices, not device breakpoints.

### Fri (Aug 29) — Post 3 (Progressive Enhancement)
- Wrote **Progressive Enhancement: Baseline First, Layers Second**.
- Baseline: usable HTML list + `noscript` notes.
- Layered: vanilla JS filters/sort; **HTMX** partial swap demo; **D3** enhancement gauge that detects HTMX swaps at runtime.

**Takeaway**
- Progressive enhancement isn’t nostalgia—it’s resiliency and better UX on slow/blocked contexts.

### Sat (Aug 30) — Polish & hygiene
- Filled `robots.txt`; double-checked Open Graph and JSON-LD on all articles.
- Verified relative links under project-page path.
- Quick Lighthouse passes (local) to keep perf/a11y in shape.

---

## What I learned (the short list)

- **Semantics drive everything:** headings, landmarks, and real controls reduce JS and improve both a11y and SEO.
- **Fluid first, breakpoints later:** `clamp()` + Grid yield layouts that breathe; add container queries where components need them.
- **Enhance, don’t replace:** baseline stays useful; HTMX/JS/D3 are optional layers.
- **CI over convenience:** Pages hosts; CI builds. Lockfile makes `npm ci` reproducible.
- **Small, local visuals:** Each post owns its D3 code; no global monolith.

---

## Decisions I’m happy with

- **Multipage HTML** (no SPA) → great crawlability, simpler mental model.
- **Page-local D3** → tiny, focused, and measurable.
- **Relative URLs** → zero base-href headaches under project pages.
- **Tailwind + purge** → consistent styling without a bespoke CSS framework.

---

## Things I’d change if I had more time

- Add a **Markdown prebuild** to generate `public/content/index.json`, list/teaser partials, and RSS.
- Expand the **a11y self-audit** with color contrast checks and landmark labeling coverage.
- Add **unit tests** for DOM checks (e.g., using a tiny headless runner) and a pre-commit hook for broken links.

---

## Known limitations

- No search yet; lists are static HTML + HTMX partials.
- No service worker/offline bundling.
- D3 is loaded per-page via CDN right now; could switch to local bundling if I consolidate charts.

---

## Quick commands

```bash
# from weeks/W04_20250825_web-blog/site/
npm install
npm run build
````

Push to `main` to redeploy via CI.

---

## Detach plan (if I spin this out later)

* Copy `site/` as repo root; keep the same build scripts.
* Keep the GH workflow (publish `public/`).
* Optionally add a content pipeline (MD → HTML/partials → RSS) and a tiny search index.

---

## Done / Next

**Done**

* [x] Scaffold + CI + GH Pages
* [x] Semantic article + self-audit (D3)
* [x] Responsive article + CPL probe (D3)
* [x] Progressive enhancement article + HTMX + D3 gauge

**Next**

* [ ] D3.js interative elements (pre-compiled for static site)
* [ ] MD → HTML content generator + `content/index.json`
* [ ] RSS + sitemap.xml
* [ ] Color-contrast audit + fixes
* [ ] Benchmarks page ported from Week-2 with interactive toggles

---

## One-liner (why this week matters)

I shipped a clean, **evidence-backed** static blog: accessible by default, responsive by design, and enhanced where it pays—ready to showcase work without hiding behind frameworks.
