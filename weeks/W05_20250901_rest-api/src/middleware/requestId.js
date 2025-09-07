import { randomUUID } from "crypto";
export function requestId(req, _res, next) {
  req.id = randomUUID();
  next();
}
