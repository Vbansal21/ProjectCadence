const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
export const validateUuidParam = (paramName = "id") => (req, res, next) => {
  const v = req.params[paramName];
  if (!UUID_RE.test(v)) return res.status(400).json({ error: `Invalid ${paramName}` });
  next();
};
