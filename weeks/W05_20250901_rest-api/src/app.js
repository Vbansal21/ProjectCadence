import express from "express";
import morgan from "morgan";
import helmet from "helmet";
import cors from "cors";
import usersRouter from "./routes/users.routes.js";
import { notFound } from "./middleware/notFound.js";
import { errorHandler } from "./middleware/errorHandler.js";import swaggerUi from "swagger-ui-express";
import spec from "./docs/openapi.json" with { type: "json"};

const app = express();

app.use(helmet());
app.use(cors());
app.use(express.json());
app.use(morgan("dev"));

// Root index (simple API metadata)
app.get("/", (_req, res) => {
  res.json({
    name: "Week-05 REST API",
    version: "0.1.0",
    endpoints: ["/health", "/users", "/docs"]
  });
});

app.get("/favicon.ico", (_req, res) => res.status(204).end());

app.get("/health", (_req, res) => res.json({ ok: true }));

app.use("/users", usersRouter);

app.use("/docs", swaggerUi.serve, swaggerUi.setup(spec));

app.use(notFound);
app.use(errorHandler);

export default app;
