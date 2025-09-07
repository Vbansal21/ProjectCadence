import request from "supertest";
import app from "../src/app.js";

test("400 on createArticle with invalid authorId", async () => {
  const res = await request(app)
    .post("/v1/articles")
    .set("x-api-key", process.env.API_KEY)
    .send({ title: "Hello", body: "World", authorId: "00000000-0000-4000-8000-000000000000" });
  expect(res.status).toBe(400);
  expect(res.body.error).toMatch(/Invalid authorId/);
});

test("404 on get non-existent article", async () => {
  const res = await request(app).get("/v1/articles/00000000-0000-4000-8000-000000000000");
  expect(res.status).toBe(404);
});

test("listArticles supports authorId filter + pagination", async () => {
  const u = await request(app).post("/v1/users").send({ name: "Al", email: "a@a.com" });
  const id = u.body.data.id;
  for (let i = 0; i < 5; i++) {
    await request(app)
      .post("/v1/articles")
      .set("x-api-key", process.env.API_KEY)
      .send({ title: `Title ${i}`, body: "Body___", authorId: id });
  }
  const list = await request(app).get(`/v1/articles?authorId=${id}&page=1&limit=2&sort=createdAt:asc`);
  expect(list.status).toBe(200);
  expect(list.body.meta).toEqual(expect.objectContaining({ page: 1, limit: 2, total: 5 }));
  expect(list.body.data).toHaveLength(2);
});
