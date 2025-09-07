import request from "supertest";
import app from "../src/app.js";

test("create user -> create article -> list", async () => {
  const u = await request(app).post("/users").send({ name: "Lin", email: "lin@example.com" });
  const userId = u.body.data.id;

  const a = await request(app)
  .post("/articles")
  .set("x-api-key", "dev-123")
  .send({
    title: "Hello",
    body: "World",
    authorId: userId
  });
  expect(a.status).toBe(201);

  const list = await request(app).get("/articles");
  expect(list.status).toBe(200);
  expect(list.body.data.length).toBe(1);
});
