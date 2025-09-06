export const errorHandler = (err, _req, res, _next) => {
    // Joi validation error
    if (err && err.isJoi) {
      return res.status(400).json({
        error: "ValidationError",
        details: err.details.map(d => ({ message: d.message, path: d.path }))
      });
    }
    const status = err.status || 500;
    const message = err.message || "Internal Server Error";
    res.status(status).json({ error: message });
  };
