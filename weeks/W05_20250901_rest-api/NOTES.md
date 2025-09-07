# NOTES.md — Week 05 · RESTful API in Node/Express (No Generators)

**Window:** 31 Aug - 06 Sep 2025
**Scope:** Build a small-but-real REST API from scratch (no scaffolding), cover routing/middleware/error handling/validation/testing/docs, and ship a lightweight in-repo console to poke the endpoints.

---

## 1) Intent

I wanted a minimal, *understand-every-line* backend: Express with **ESM** modules, **versioned routes**, consistent responses, **Joi** validation, **API-key** write protection, **rate limiting**, **request IDs**, **Swagger UI**, and **Jest + Supertest** tests. No database this week; in-memory stores only. A tiny HTML console (`public/index.html`) makes the API tangible.

---

## 2) What happened (chronology with pain points)

1. **ESM + JSON imports.**

   * Hit `ERR_IMPORT_ATTRIBUTE_MISSING` importing `openapi.json`.
   * Settled on **dynamic import with import attributes**:

     ```js
     const mod = await import("./docs/openapi.json", { with: { type: "json" } });
     const spec = mod.default;
     ```

   * This requires **Node ≥ 22**. I pinned engines and verified locally (Node 23 worked).

2. **Jest with ESM.**

   * Jest initially choked on `import`.
   * Fixed by running with:

     ```sh
     NODE_OPTIONS=--experimental-vm-modules
     ```

     and keeping tests pure ESM. No Babel.

3. **Validation vs controller logic.**

   * Early error tests were catching **Joi**'s `ValidationError` before controller branches (e.g., "Invalid authorId").
   * I adjusted tests to send *valid* payloads that reach the controller and then throw **custom `HttpError`**.

4. **Auth semantics in tests.**

   * My first API-key middleware skipped auth in `NODE_ENV=test`. That broke the "401 without key" test once `API_KEY` was set.
   * Final rule: **if a key is configured, enforce it** (even in tests).

5. **Versioning.**

   * Moved routers under `/v1`. Updating tests and the console "API Base" field avoided 404s.

6. **Param hygiene.**

   * Added a UUID v4 guard to every `/:id` route to fail fast with `400` on malformed IDs.

7. **Console UI.**

   * Served `public/` via `express.static`.
   * Used **Helmet** with a CSP that allows my inline script (scoped, still conservative). The console has `API Base` (defaults `/v1`) and an `x-api-key` field.

---

## 3) Decisions and rationale

* **ESM everywhere**: aligns with modern Node; fewer toolchain crutches.
* **Dynamic JSON import**: keeps Swagger out of the test path and avoids bundling quirks.
* **Uniform responses**: `ok()/created()` helpers keep payloads predictable and reduce test noise.
* **Opt-in security**: write routes require `x-api-key` **only** when `API_KEY` is set; convenient locally, secure in prod.
* **Versioned base `/v1`**: lets me evolve without breaking consoles/clients later.
* **In-memory stores**: intentionally simple; the API surface is the focus. Postgres comes next.

---

## 4) Architecture (short)

```
public/index.html  →  / (static)
Swagger UI         →  /docs       (non-test env)
Versioned API      →  /v1/*
Routers: users, articles, comments
Controllers → in-memory stores (Map-backed)
Middleware: requestId, morgan, helmet, cors, rateLimit, validate(Joi), validateUuidParam, apiKey
Error path: HttpError | Joi → errorHandler (uniform {error, rid})
```

---

## 5) Key patterns I want to remember

### 5.1 Response helpers and typed errors

```js
// utils/http.js
export class HttpError extends Error {
  constructor(status, message) { super(message); this.status = status; }
}

export const ok      = (res, data, meta) => res.status(200).json(meta ? { data, meta } : { data });
export const created = (res, data, meta) => res.status(201).json(meta ? { data, meta } : { data });
```

```js
// middleware/errorHandler.js (shape normalized)
export function errorHandler(err, req, res, _next) {
  const rid = req.id;
  if (err.isJoi) {
    return res.status(400).json({ error: "ValidationError", rid, details: err.details?.map(d => d.message) });
  }
  if (err instanceof Error && "status" in err) {
    return res.status(err.status).json({ error: err.message, rid });
  }
  return res.status(500).json({ error: "InternalError", rid });
}
```

### 5.2 API-key enforcement (final form)

```js
// middleware/apiKey.js
import { config } from "../config.js";
export function apiKey() {
  const key = config.apiKey;                   // empty → auth off
  return (req, res, next) => {
    if (!key) return next();                   // skip only when no key configured
    const provided = req.header("x-api-key") || "";
    if (provided === key) return next();
    return res.status(401).json({ error: "Unauthorized" });
  };
}
```

### 5.3 UUID guard

```js
// middleware/validateParam.js
const UUIDv4 = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
export const validateUuidParam = (name="id") => (req, res, next) =>
  UUIDv4.test(req.params[name]) ? next() : res.status(400).json({ error: `Invalid ${name}` });
```

### 5.4 List queries (paging/sort + filter)

```js
// controllers/articles.controller.js (excerpt)
import { parseQuery, applyListQuery } from "../utils/query.js";
import { ok, created, HttpError } from "../utils/http.js";
import { db } from "../models/db.js";

export async function listArticles(req, res) {
  const all = db.articles.all();
  const filtered = req.query.authorId ? all.filter(a => a.authorId === req.query.authorId) : all;
  const q = parseQuery(req.query);
  const { slice, meta } = applyListQuery(filtered, q);
  return ok(res, slice, meta);
}

export async function createArticle(req, res) {
  const author = db.users.get(req.validated.authorId);
  if (!author) throw new HttpError(400, "Invalid authorId");
  return created(res, db.articles.insert(req.validated));
}
```

### 5.5 Deterministic tests

```js
// tests/setup.js
import { db } from "../src/models/db.js";
beforeEach(() => db.resetAll());   // Map.clear() on every store
```

```json
// package.json (relevant bits)
{
  "engines": { "node": ">=22 <24" },
  "scripts": {
    "dev": "node --watch src/server.js",
    "start": "node src/server.js",
    "test": "cross-env NODE_OPTIONS=--experimental-vm-modules NODE_ENV=test API_KEY=dev-123 jest --runInBand",
    "lint": "eslint src tests public --ext .js"
  }
}
```

---

## 6) The tiny console (UX notes)

* Single page (`public/index.html`) with dark UI, inputs for **API Base** (defaults to `/v1`) and **x-api-key**.
* Uses `fetch` with automatic JSON and key header injection.
* Keeps local lookup maps (userId → name, articleId → title) so tables read well.
* Helmet CSP: allows `'self'` and `'unsafe-inline'` (script) to keep the file self-contained. I'll tighten CSP once I externalize JS.

---

## 7) Testing surface (what I covered)

* **Users:** create + list happy path.
* **Articles:** create (with key), list with `authorId` filter + pagination, 404 on delete missing, 400 invalid UUID.
* **Comments:** create (with/without author), list filter by `articleId`, update/get/delete.
* **Security:** 401 when key is configured but header missing.
* **500 path:** forced error to exercise the generic handler.
* CI runs **only** for the Week-5 path; I added ESLint before tests.

---

## 8) What I learned / reinforced

* ESM in Node is fine now; the rough edges are mostly around test runners and JSON import attributes.
* Validation order matters: if I want controller-level errors tested, I must satisfy schema first.
* A small **response contract** pays off quickly in tests and docs.
* Versioning up front saves churn (tests + console only needed a base tweak).
* Even a trivial console is worth it—the API feels "real" without leaving the repo.

---

## 9) What I didn't do (by design, week-scoped)

* No database; stores are Maps.
* No token-based auth or sessions; just API-key gating on writes.
* No Docker.
* No heavy observability; just request IDs and morgan.

---

## 10) Next steps (Week 06+)

* **PostgreSQL** integration: schema for users/articles/comments, migrations, a tiny data-access layer, and porting controllers.
* **Auth**: JWT for user actions, move API-key to admin/ops only.
* **Caching/E-Tags**: add conditional GET, possibly `If-None-Match`.
* **Pagination**: shift to cursor-based tokens for large lists.
* **Ablation**: mirror `/v1/users` in **Fastify**, compare with `autocannon`.

---

## 11) Quick smoke (cheat sheet)

```sh
# start
npm ci && npm run dev
# create a user (no key required)
curl -s -H 'content-type: application/json' \
  -d '{"name":"Ada Lovelace","email":"ada@compute.org"}' \
  http://localhost:3000/v1/users | jq
# create an article (needs x-api-key if API_KEY is set)
curl -s -H 'x-api-key: dev-123' -H 'content-type: application/json' \
  -d '{"title":"Hello","body":"World","authorId":"<uuid>"}' \
  http://localhost:3000/v1/articles | jq
# docs
open http://localhost:3000/docs
# console
open http://localhost:3000/
```

---

## 12) Closing thought

The API now has a clean spine: versioned surface, tight error shapes, small security hooks, a console to feel interactions, and tests to keep me honest. That's a good substrate for Week 06's database move without re-learning fundamentals.

(**P.S. :- Use of CHATGPT for learning, and formatting this note and readme file[s] - has been done extensively.**)
