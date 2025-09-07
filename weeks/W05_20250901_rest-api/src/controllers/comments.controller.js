import { db } from "../models/db.js";
import { parseQuery, applyListQuery } from "../utils/query.js";
import { ok, created, HttpError} from "../utils/http.js";

export async function listComments(req, res) {
  const all = db.comments.all();
  const filtered = req.query.authorId
    ? all.filter(a => a.authorId === req.query.authorId)
    : all;
  const q = parseQuery(req.query);
  const {slice, meta} = applyListQuery(filtered, q);
  return ok(res, slice, meta);
}
export async function createComment(req, res) {
  const { articleId, authorId } = req.validated;
  const article = db.articles.get(articleId);
  if (!article) throw new HttpError(400, "Invalid articleId");
  if (authorId && !db.users.get(authorId)) {
    throw new HttpError(400, "Invalid authorId");
  }
  const doc = db.comments.insert(req.validated);
  return created(res, doc);
}
export async function getComment(req, res) {
  const doc = db.comments.get(req.params.id);
  if (!doc) throw new HttpError(404, "Comment not found");
  return ok(res, doc);
}
export async function updateComment(req, res) {
  const doc = db.comments.update(req.params.id, req.validated);
  if (!doc) throw new HttpError(404, "Comment not found");
  return ok(res, doc);
}
export async function removeComment(req, res) {
  const ok = db.comments.remove(req.params.id);
  if (!ok) throw new HttpError(404, "Comment not found");
  return res.status(204).send();
}
