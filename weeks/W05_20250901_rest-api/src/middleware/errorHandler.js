import { HttpError } from "../utils/http.js";

export const errorHandler = (err, req, res, _next) => {
  if (err && err.isJoi) {
    return res.status(400).json({
      error: "ValidationError",
      rid: req.id,
      details: err.details.map(d => ({ message: d.message, path: d.path }))
    });
  }
  if (err instanceof HttpError) {
    return res.status(err.status).json({ error: err.message, rid: req.id });
  }
  const status = err.status || 500;
  res.status(status).json({ error: err.message || "Internal Server Error", rid: req.id });
};
