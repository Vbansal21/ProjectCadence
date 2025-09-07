import request from "supertest";
import app from "../src/app.js";

test("401 on protected route without x-api-key", async () => {
  const u = await request(app).post("/v1/users").send({ name: "Al", email: "a@a.com" });
  const res = await request(app).post("/v1/articles").send({ title: "Ttt", body: "B", authorId: u.body.data.id });
  expect(res.status).toBe(401);
});
