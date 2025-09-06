import { randomUUID } from "crypto";

function createStore() {
  const items = new Map();
  return {
    all: () => Array.from(items.values()),
    get: (id) => items.get(id) || null,
    insert: (payload) => {
      const now = new Date().toISOString();
      const doc = { id: randomUUID(), createdAt: now, updatedAt: now, ...payload };
      items.set(doc.id, doc);
      return doc;
    },
    update: (id, patch) => {
      const found = items.get(id);
      if (!found) return null;
      const updated = { ...found, ...patch, updatedAt: new Date().toISOString() };
      items.set(id, updated);
      return updated;
    },
    remove: (id) => items.delete(id)
  };
}

export const db = {
  users: createStore(),
  // articles: createStore(),
  // comments: createStore()
};
