import { config } from "../config.js";

// Middleware to enforce API key in "x-api-key" header
export function apiKey() {
  const key = config.apiKey;        // undefined means "auth off"
  return (req, res, next) => {
    // Skip enforcement if no key configured or during tests
    if (!key || process.env.NODE_ENV === "test") return next();

    const provided = req.header("x-api-key") || "";
    if (provided === key) return next();
    return res.status(401).json({ error: "Unauthorized" });
  };
}
