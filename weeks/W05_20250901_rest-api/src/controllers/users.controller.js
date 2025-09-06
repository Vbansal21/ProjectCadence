import { db } from "../models/db.js";

export async function listUsers(_req, res) {
  res.json({ data: db.users.all() });
}

export async function createUser(req, res) {
  const user = db.users.insert(req.validated);
  res.status(201).json({ data: user });
}

export async function getUser(req, res) {
  const user = db.users.get(req.params.id);
  if (!user) return res.status(404).json({ error: "User not found" });
  res.json({ data: user });
}

export async function updateUser(req, res) {
  const updated = db.users.update(req.params.id, req.validated);
  if (!updated) return res.status(404).json({ error: "User not found" });
  res.json({ data: updated });
}

export async function removeUser(req, res) {
  const ok = db.users.remove(req.params.id);
  if (!ok) return res.status(404).json({ error: "User not found" });
  res.status(204).send();
}
