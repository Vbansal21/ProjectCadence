export function apiKey(required = true) {
    const key = process.env.API_KEY || "";
    return (req, res, next) => {
      const provided = req.header("x-api-key") || "";
      if (!required && !key) return next();
      if (key && provided === key) return next();
      return res.status(401).json({ error: "Unauthorized" });
    };
  }
