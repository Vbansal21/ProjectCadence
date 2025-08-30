# Week 4 – Static Blog (HTMX · Tailwind · TS · D3)

**Week** 04 - 2025-08-25 → 2025-08-31

I built a **project-page–friendly static blog** that lives entirely inside this Week-4 directory and deploys to GitHub Pages. It’s framework-free in spirit (no SPA), but modern in practice: **HTMX** for partial swaps, **Tailwind** for styles, **TypeScript** for small enhancements, and **D3.js** for focused, page-local visuals. Everything is static and relative-linked so it works under a project page path.

- **Live (project page):** served from the repo’s `gh-pages` branch.
- **Scope this week:** ship the scaffold + 3 advanced posts
  - Semantic HTML for A11y & SEO
  - Responsive Design: Fluid Type & Grid
  - Progressive Enhancement: Baseline → Layers

---

## What’s in here

```

weeks/W04\_20250825\_static-web-blog/
├─ README.md
├─ NOTES.md
└─ site/
├─ public/                  # published artifacts (GH Pages serves this)
│  ├─ index.html
│  ├─ blog/
│  │  ├─ index.html
│  │  └─ \_partials/
│  │     ├─ list.html
│  │     └─ teaser.html
│  ├─ content/
│  │  └─ index.json         # placeholder for a future content indexer
│  ├─ demos/
│  │  └─ d3-bar.html
│  ├─ assets/
│  │  ├─ css/site.css
│  │  ├─ js/site.js
│  │  ├─ js/charts/demo.js
│  │  └─ media/{favicon.svg, og-image.svg}
│  └─ robots.txt
├─ src/
│  ├─ css/site.css          # Tailwind input
│  └─ ts/
│     ├─ site.ts
│     └─ charts/demo.ts
├─ package.json
├─ tsconfig.json
├─ tailwind.config.cjs
└─ postcss.config.cjs

```

### Blog posts added this week

- `public/blog/posts/semantic-accessibility.html`
  Self-audits semantic/a11y features and renders a live **D3** chart of the audit.

- `public/blog/posts/responsive-layouts.html`
  Explains **fluid type + grid** and includes a **D3 readability probe** (characters per line) that updates on resize.

- `public/blog/posts/progressive-enhancement.html`
  Shows a **no-JS baseline**, **HTMX** partial swaps, and **vanilla JS** enrichments. Includes a **D3 enhancement gauge**.

The blog index (`public/blog/_partials/list.html`) and teaser (`teaser.html`) link to these pages.

---

## Local dev

From **`weeks/W04_20250825_static-web-blog/site/`**:

```bash
# first time
npm install

# build Tailwind + TypeScript bundles
npm run build
````

Open `public/index.html` in a simple local server (e.g., VS Code Live Server). All links are relative; no special base path needed.

> Why npm if it’s GitHub Pages?
> Pages only hosts static files. I compile Tailwind/TS with npm **in CI** so Pages serves minified static assets.

---

## Deploy

CI workflow lives at repo root:

```
.github/workflows/deployW04.yml
```

* Triggers on changes under `weeks/W04_20250825_static-web-blog/site/**`
* Runs `npm ci && npm run build`
* Publishes `site/public/` to `gh-pages`
* Repo settings: Pages → Source = `gh-pages` (branch), Folder = `/`

No secrets required; uses the repo `GITHUB_TOKEN` with `contents: write`.

---

## How to add a new post (static)

1. Create `site/public/blog/posts/<slug>.html` (semantic HTML, relative paths).
2. Add a card link in `site/public/blog/_partials/list.html` and (optionally) `teaser.html`.
3. If the post includes D3, prefer **page-local scripts** and inline `fill`/`stroke` to avoid Tailwind purge collisions.
4. Commit + push to `main` → CI builds and redeploys.

> Future: a tiny prebuild can convert Markdown → HTML and generate `public/content/index.json` for filtering and RSS. The scaffold already includes `content/index.json`.

---

## Accessibility & SEO practices I enforced

* Landmarks: `header`, `nav[aria-label]`, `main`, `footer`.
* Skip-to-content link, visible focus rings.
* Logical heading outline; no fake headings.
* Purposeful alt text; decorative images use `alt=""`.
* Articles include title/description, Open Graph/Twitter, and `BlogPosting` JSON-LD.
* Multipage site (no SPA router) → good crawlability and deep-linking.

---

## Performance budget

* Keep non-interactive pages ≤ \~120 KB JS and ≤ \~80 KB CSS (post-purge Tailwind).
* Load **D3** only on pages that need it (each article scopes its chart).
* Everything is static; no client-side routing.

---

## Monorepo + detach-ready

* All site code lives under this Week-4 directory with relative links → it’s trivially extractable to a standalone repo later.
* The GitHub Actions workflow publishes **only this subdir’s `public/`** to `gh-pages`.

---

## Résumé snippet

> Built and deployed a responsive, accessible static blog (HTMX, Tailwind, TypeScript, D3) on GitHub Pages. Implemented self-auditing posts (semantic/a11y, readability, enhancement readiness) with page-local D3 visuals. CI builds and publishes from a subfolder; all content remains framework-free and crawlable.
