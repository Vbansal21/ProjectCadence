import { db } from "../models/db.js";
import { ok, created, HttpError } from "../utils/http.js";

export async function listUsers(_req, res) {
  return ok(res, db.users.all());
}

export async function createUser(req, res) {
  const user = db.users.insert(req.validated);
  return created(res, user);
}

export async function getUser(req, res) {
  const user = db.users.get(req.params.id);
  if (!user) throw new HttpError(404, "User not found");
  return ok(res, user);
}

export async function updateUser(req, res) {
  const updated = db.users.update(req.params.id, req.validated);
  if (!updated) throw new HttpError(404, "User not found");
  return ok(res, updated);
}

export async function removeUser(req, res) {
  const okDelete = db.users.remove(req.params.id);
  if (!okDelete) throw new HttpError(404, "User not found");
  return res.status(204).send();
}
