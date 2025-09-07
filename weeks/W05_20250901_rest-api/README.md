# Week-05 — RESTful API in Node/Express (No Generators)

**Week** 05 - 2025-09-01 → 2025-09-07

> A from-scratch Express service with clean architecture, versioned routes, consistent responses, input validation, auth via API key, rate limiting, OpenAPI docs, a tiny in-repo API Console (HTML/CSS/JS), and Jest + Supertest tests. All built without scaffolding generators.

---

## Introduction

This project implements a production-lean REST API focusing on fundamentals—routing, middleware, error handling, request validation, deterministic tests, and minimal documentation tooling—while keeping the data layer in-memory for simplicity. The code is idiomatic **ESM** (native `import`) and **JSON import attributes**.

---

## Context

* **Track:** Week 5 of the 52-Weeks/52-Projects cadence.
* **Repo path (monorepo):** `weeks/W05_20250901_rest-api/`
* **Runtime:** Node ≥ **22** (tested on 23.x), Express, Joi, Swagger-UI.
* **Constraints:** No code generators; deterministic, minimal surface; optional small frontend console.
* **Security posture:** Optional write-protection using an API key header, basic rate limiting, request IDs, Helmet CSP tuned for the static console.

---

## Objectives

1. Deliver a **versioned** REST API with users, articles, comments.
2. Enforce **validation** and **uniform responses**; centralize error handling.
3. Provide **auth** via `x-api-key` (opt-in by env var), **rate limit**, **request logging**, **request IDs**.
4. Ship **OpenAPI** docs and a **tiny API Console** for quick manual exercises.
5. Cover core paths with **Jest + Supertest** (deterministic, fast).
6. Prepare for future DB integration (Week 6) and optional Fastify ablation.

---

## Methodology

### Architecture at a glance

```mermaid
flowchart LR
  A[Client / API Console] -->|HTTP JSON| B[/v1 Router]
  B --> C[Route Modules]
  C --> D[Controllers]
  D --> E[In-Memory Stores]
  B --> M[Middleware: auth, validate, rate, id, logging]
  M --> H[Error Handler]
  B --> S[SwaggerUI /docs]
  A -->|GET static| T[(public/ index.html)]
```

### Response Contract

* **Success:**

  ```json
  { "data": <payload>, "meta": { "page":1, "limit":10, "total": 42 } }   // meta optional
  ```
* **Error:**

  ```json
  { "error": "User not found", "rid": "<uuid>" }
  ```

### Versioning

* All API routes are mounted under **`/v1`**.
* Static console is served from `/` and defaults to `/v1` as API base.

### Validation, Errors, Helpers

* **Joi** schemas validate request bodies.
* Controllers use:

  * `ok(res, data, meta?)` (200)
  * `created(res, data, meta?)` (201)
  * `throw new HttpError(status, message)` for error branches (e.g., 400/404).
* Central `errorHandler` maps Joi and `HttpError` to consistent error shapes.

### Security & QoL

* **API key** (opt-in): set `API_KEY` env; writes require header `x-api-key`.
* **Rate limit:** default `120 req / 60 s` (configurable).
* **Helmet** with a small CSP allowing the in-repo console.
* **Request IDs** (UUID v4) for traceability in logs and error payloads.
* **morgan** logging with request ID token.

---

## Results

### Features delivered

* **Resources:** `users`, `articles`, `comments` (CRUD where appropriate).
* **Filtering & Paging:**

  * `GET /v1/articles?authorId=<uuid>&page=1&limit=10&sort=createdAt:asc`
  * `GET /v1/comments?articleId=<uuid>&authorId=<uuid>`
* **UUID guards:** all `/:id` routes return **400** on malformed UUID.
* **OpenAPI UI:** `/docs` (mounted only outside of tests).
* **Static Console:** `/` → interactive HTML UI for CRUD (see below).
* **CI:** workflow scoped to this week's directory; lint + tests.

### What's intentionally **not** in scope

* Real database (will land in Week 6).
* Authentication schemes beyond API key.
* Docker image (optional stretch).

---

## Project Layout

```text
W05_20250901_rest-api/
├─ public/                 # API Console (index.html, CSS/JS inline)
├─ src/
│  ├─ routes/              # users.routes.js, articles.routes.js, comments.routes.js
│  ├─ controllers/         # users.controller.js, articles.controller.js, comments.controller.js
│  ├─ middleware/          # validate.js, validateParam.js, apiKey.js, requestId.js, errorHandler.js
│  ├─ models/              # db.js (in-memory stores with resetAll())
│  ├─ schemas/             # Joi schemas: users, articles, comments
│  ├─ utils/               # http.js (HttpError, ok, created), query.js (paging/sort helpers)
│  ├─ docs/                # openapi.json (served at /docs in non-test env)
│  ├─ config.js            # env-driven constants (port, rate, cors, apiKey)
│  ├─ app.js               # pipeline, versioned router, static hosting
│  └─ server.js            # bootstrap (reads PORT), trust proxy
├─ tests/                  # Jest + Supertest suites
├─ package.json            # engines.node >=22, scripts (dev/start/test/lint)
├─ .eslintrc.json          # minimal ESLint config
└─ .github/workflows/week5-ci.yml
```

---

## Running Locally

### Prerequisites

* **Node ≥ 22** (supports `import ... with { type: "json" }`)
* `npm` or `pnpm`/`yarn` (examples use npm)

### Quickstart

```bash
npm ci
npm run dev            # or: npm start
# default: http://localhost:3000
```

### Environment

Create `.env` (optional; example):

```ini
NODE_ENV=development
PORT=3000
API_KEY=dev-123
```

> If `API_KEY` is set, **POST/PUT/DELETE** require header `x-api-key: <value>`.

### Static Console (UI)

Open `http://localhost:3000/`.
Top bar:

* **API Base** defaults to `/v1` (change if needed).
* **x-api-key** text box (leave empty if no key configured).
* Live tables for Users, Articles, Comments; add/update/delete via buttons.

---

## API Reference (selected)

> Prefix all paths with **`/v1`**.

### Users

* **POST** `/users` — create

  ```json
  { "name": "Ada Lovelace", "email": "ada@compute.org" }
  ```

  * 201 → `{ "data": { "id": "<uuid>", "name": "...", "email": "...", "createdAt": "...", "updatedAt": "..." } }`
  * 400 (Joi)

* **GET** `/users` — list (no paging yet for users; simple array)

  * 200 → `{ "data": [User, ...] }`

* **GET** `/users/{id}` — get

  * 200 → `{ "data": User }`
  * 400 invalid UUID / 404 not found

* **PUT** `/users/{id}` — update partial (per schema)

  * 200 → `{ "data": User }`
  * 400 invalid UUID or validation / 404

* **DELETE** `/users/{id}`

  * 204 (no body) or 400 / 404

### Articles

* **GET** `/articles?authorId&page&limit&sort=field:asc|desc`

  * 200 → `{ "data": Article[], "meta": { "page":..., "limit":..., "total":... } }`

* **POST** `/articles` *(write requires API key if configured)*

  ```json
  { "title": "Hello", "body": "World", "authorId": "<uuid>" }
  ```

  * 201 → `{ "data": Article }`
  * 400 (validation or invalid `authorId`)

* **GET/PUT/DELETE** `/articles/{id}`

  * 200/204 or 400/404

### Comments

* **GET** `/comments?articleId=<uuid>&authorId=<uuid>&page&limit`

  * 200 → `{ "data": Comment[], "meta": {...} }`

* **POST** `/comments` *(write requires API key if configured)*

  ```json
  { "articleId": "<uuid>", "text": "great post!", "authorId": "<uuid|optional>" }
  ```

  * 201 → `{ "data": Comment }`
  * 400 invalid `articleId` or `authorId`

* **GET/PUT/DELETE** `/comments/{id}`

  * as above

---

## OpenAPI / Swagger

* Spec file: `src/docs/openapi.json`
* Served at: **`/docs`** (only when `NODE_ENV !== "test"`)
* `servers[0].url` points to `/v1`
* Components: `User`, `Article`, `Comment` (+ corresponding Create models), `Error`, and `apiKey` security scheme

---

## Testing

### Commands

```bash
npm test
```

* Jest runs in-band with **ESM** enabled via `NODE_OPTIONS=--experimental-vm-modules`.
* Test env sets `API_KEY=dev-123` to exercise 401 branches.
* `tests/setup.js` calls `db.resetAll()` to keep runs deterministic.

### What's covered

* Happy paths for users and articles
* Error cases:

  * 400 (validation), 400 (invalid UUID param), 401 (missing API key on writes), 404 (not found), and a 500 smoke path
* Filters + pagination (articles, comments)

---

## Configuration

`src/config.js` centralizes:

* `env`, `port`, `apiKey`
* `cors` (defaults `origin: true, credentials: true`)
* `rate` (defaults `windowMs: 60_000, max: 120`)

Trusts proxy: `app.set("trust proxy", 1)` (works behind Render/Railway/Heroku).

---

## Development Notes

* **ESM & JSON import attributes:** Node ≥ 22 is required for `await import("./openapi.json", { with: { type: "json" } })`.
* **CSP:** Helmet allows inline script for the console (`'unsafe-inline'`)—tighten when hosting the UI elsewhere.
* **Versioning:** All new endpoints should mount under `/v1`. Add `/v2` later without breaking existing clients.

---

## CI

Minimal GitHub Actions workflow only for this week's path:

* `npm ci` → `npm run lint` → `npm test`
* Node 20+ runner (prefer 22 to mirror production features)

Edit: `.github/workflows/week5-ci.yml`

---

## Optional: Ablation (Express vs Fastify)

### Quick RPS probe with autocannon

```bash
npm i -D autocannon
npx autocannon -c 50 -d 15 http://localhost:3000/v1/users
```

*(Add a minimal Fastify mirror under `experiments/fastify/` and replicate `/v1/users` to make a fair run. Keep identical logic & validation for apples-to-apples.)*

---

## Future Work (Week 6+)

* Replace in-memory stores with **PostgreSQL** (via `pg` or a lightweight ORM).
* Add **JWT** sessions for user auth; role-based permissions.
* **Caching** and **E-Tags** on list endpoints; conditional GETs.
* **Rate limiting** per API key; basic quota metrics.
* **Better pagination** (cursor/opaque tokens).
* Hardening: input size limits, timeout/abort handling, structured logs.

---

## Quick Commands (Cheat-Sheet)

```bash
# Run
npm ci && npm run dev

# Lint + test
npm run lint && npm test

# Create a user (no API key needed)
curl -s -H "content-type: application/json" \
  -d '{"name":"Ada Lovelace","email":"ada@compute.org"}' \
  http://localhost:3000/v1/users | jq

# Create an article (API key required if configured)
curl -s -H "x-api-key: dev-123" -H "content-type: application/json" \
  -d '{"title":"Hello","body":"World","authorId":"<uuid>"}' \
  http://localhost:3000/v1/articles | jq

# Docs
open http://localhost:3000/docs
```

---

## References

* Express.js (official)
* Helmet (security headers)
* Joi (validation)
* Jest & Supertest (testing)
* OpenAPI / Swagger UI

*(Official docs are widely available; keep versions aligned with your `package.json`.)*

---

## Conclusion

The Week-05 API delivers a clean, versioned, validated, and test-covered REST foundation with a minimal console and OpenAPI docs.
