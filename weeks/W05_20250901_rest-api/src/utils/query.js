export function parseQuery(q) {
    const page = Math.max(1, Number(q.page) || 1);
    const limit = Math.min(100, Math.max(1, Number(q.limit) || 20));
    const sort = (q.sort || "createdAt:desc").split(":");
    return { page, limit, sort: { field: sort[0], dir: (sort[1] || "desc").toLowerCase() } };
  }

  export function applyListQuery(items, { page, limit, sort }) {
    const arr = [...items].sort((a, b) => {
      const av = a[sort.field], bv = b[sort.field];
      if (av === bv) return 0;
      return (av > bv ? 1 : -1) * (sort.dir === "desc" ? -1 : 1);
    });
    const start = (page - 1) * limit;
    const slice = arr.slice(start, start + limit);
    return { slice, meta: { page, limit, total: arr.length } };
  }
