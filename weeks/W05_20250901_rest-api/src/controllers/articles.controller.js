import { db } from "../models/db.js";
import { applyListQuery, parseQuery } from "../utils/query.js";

export async function listArticles(req, res) {
  const all = db.articles.all();
  const { page, limit, sort } = parseQuery(req.query);
  const { slice, meta } = applyListQuery(all, { page, limit, sort });
  res.json({ data: slice, meta });
}

export async function createArticle(req, res) {
  // basic referential integrity: author must exist
  const author = db.users.get(req.validated.authorId);
  if (!author) return res.status(400).json({ error: "Invalid authorId" });

  const doc = db.articles.insert(req.validated);
  res.status(201).json({ data: doc });
}
export async function getArticle(req, res) {
  const doc = db.articles.get(req.params.id);
  if (!doc) return res.status(404).json({ error: "Article not found" });
  res.json({ data: doc });
}
export async function updateArticle(req, res) {
  const doc = db.articles.update(req.params.id, req.validated);
  if (!doc) return res.status(404).json({ error: "Article not found" });
  res.json({ data: doc });
}
export async function removeArticle(req, res) {
  const ok = db.articles.remove(req.params.id);
  if (!ok) return res.status(404).json({ error: "Article not found" });
  res.status(204).send();
}
