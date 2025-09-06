import request from "supertest";
import app from "../src/app.js";

test("POST /users then GET /users", async () => {
  const create = await request(app)
    .post("/users")
    .send({ name: "Ada Lovelace", email: "ada@compute.org" });
  expect(create.statusCode).toBe(201);
  expect(create.body.data).toHaveProperty("id");

  const list = await request(app).get("/users");
  expect(list.statusCode).toBe(200);
  expect(Array.isArray(list.body.data)).toBe(true);
  expect(list.body.data.length).toBe(1);
});
