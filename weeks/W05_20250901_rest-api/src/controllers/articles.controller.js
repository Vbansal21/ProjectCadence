import { db } from "../models/db.js";
import { applyListQuery, parseQuery } from "../utils/query.js";
import { ok, created, HttpError } from "../utils/http.js";

export async function listArticles(req, res) {
  const all = db.articles.all();
  const filtered = req.query.authorId
    ? all.filter(a => a.authorId === req.query.authorId)
    : all;
  const q = parseQuery(req.query);
  const {slice, meta} = applyListQuery(filtered, q);
  return ok(res, slice, meta);
}

export async function createArticle(req, res) {
  const author = db.users.get(req.validated.authorId);
  if (!author) throw new HttpError(400, "Invalid authorId");
  const doc = db.articles.insert(req.validated);
  return created(res, doc);
}
export async function getArticle(req, res) {
  const doc = db.articles.get(req.params.id);
  if (!doc) throw new HttpError(404, "Article not found");
  return ok(res, doc);
}
export async function updateArticle(req, res) {
  const doc = db.articles.update(req.params.id, req.validated);
  if (!doc) throw new HttpError(404, "Article not found");
  return ok(res, doc);
}
export async function removeArticle(req, res) {
  const ok = db.articles.remove(req.params.id);
  if (!ok) throw new HttpError(404, "Article not found");
  return res.status(204).send();
}
