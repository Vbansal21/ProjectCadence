import { db } from "../models/db.js";

export async function listComments(_req, res) {
  res.json({ data: db.comments.all() });
}
export async function createComment(req, res) {
  const { articleId, authorId } = req.validated;
  const article = db.articles.get(articleId);
  if (!article) return res.status(400).json({ error: "Invalid articleId" });
  if (authorId && !db.users.get(authorId)) {
    return res.status(400).json({ error: "Invalid authorId" });
  }
  const doc = db.comments.insert(req.validated);
  res.status(201).json({ data: doc });
}
export async function getComment(req, res) {
  const doc = db.comments.get(req.params.id);
  if (!doc) return res.status(404).json({ error: "Comment not found" });
  res.json({ data: doc });
}
export async function updateComment(req, res) {
  const doc = db.comments.update(req.params.id, req.validated);
  if (!doc) return res.status(404).json({ error: "Comment not found" });
  res.json({ data: doc });
}
export async function removeComment(req, res) {
  const ok = db.comments.remove(req.params.id);
  if (!ok) return res.status(404).json({ error: "Comment not found" });
  res.status(204).send();
}
