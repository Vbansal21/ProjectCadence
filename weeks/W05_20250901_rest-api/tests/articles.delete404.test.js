import request from "supertest";
import app from "../src/app.js";

test("404 on delete non-existent article", async () => {
  const res = await request(app)
    .delete("/v1/articles/00000000-0000-4000-8000-000000000000")
    .set("x-api-key", process.env.API_KEY);
  expect(res.status).toBe(404);
});
