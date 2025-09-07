import express from "express";
import morgan from "morgan";
import helmet from "helmet";
import cors from "cors";
import rateLimit from "express-rate-limit";

import usersRouter from "./routes/users.routes.js";
import { notFound } from "./middleware/notFound.js";
import { errorHandler } from "./middleware/errorHandler.js";import swaggerUi from "swagger-ui-express";
import spec from "./docs/openapi.json" with { type: "json"};
import articlesRouter from "./routes/articles.routes.js";
import commentsRouter from "./routes/comments.routes.js";
import { requestId } from "./middleware/requestId.js";

const app = express();

app.use(helmet());
app.use(cors({origin: true, credentials: true }));
app.use(rateLimit({ windowMs: 60_000, max:120}))
app.use(express.json());
app.use(morgan("dev"));

morgan.token("rid", (req) => req.id);
app.use(requestId);
app.use(morgan(':method :url :status - rid=:rid - :response-time ms'));


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

app.use("/articles", articlesRouter);

app.use("/comments", commentsRouter);

app.use(notFound);
app.use(errorHandler);

export default app;
