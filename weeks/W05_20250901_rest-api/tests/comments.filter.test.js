import request from "supertest";
import app from "../src/app.js";

test("list comments filter by articleId", async () => {
  const u = await request(app).post("/v1/users").send({ name: "Al", email: "a@a.com" });
  const a = await request(app).post("/v1/articles")
    .set("x-api-key", process.env.API_KEY)
    .send({ title: "Title", body: "Body", authorId: u.body.data.id });
  const articleId = a.body.data.id;

  await request(app).post("/v1/comments").set("x-api-key", process.env.API_KEY)
    .send({ articleId, text: "c1", authorId: u.body.data.id });
  await request(app).post("/v1/comments").set("x-api-key", process.env.API_KEY)
    .send({ articleId, text: "c2" });

  const list = await request(app).get(`/v1/comments?articleId=${articleId}&limit=1`);
  expect(list.status).toBe(200);
  expect(list.body.meta.total).toBe(2);
  expect(list.body.data).toHaveLength(1);
});
