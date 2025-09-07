// src/utils/http.js
export class HttpError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}
export const ok = (res, data, meta) =>
  res.status(200).json(meta ? { data, meta } : { data });

export const created = (res, data, meta) =>
  res.status(201).json(meta ? { data, meta } : { data });
