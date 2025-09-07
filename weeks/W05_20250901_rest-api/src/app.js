import express from "express";
import morgan from "morgan";
import helmet from "helmet";
import cors from "cors";
import rateLimit from "express-rate-limit";
import { config } from "./config.js";

import usersRouter from "./routes/users.routes.js";
import { notFound } from "./middleware/notFound.js";
import { errorHandler } from "./middleware/errorHandler.js";
import swaggerUi from "swagger-ui-express";
import articlesRouter from "./routes/articles.routes.js";
import commentsRouter from "./routes/comments.routes.js";
import { requestId } from "./middleware/requestId.js";

import path from "path";
import { fileURLToPath } from "url";

const app = express();
app.set("trust proxy", 1);

// app.use(helmet());
app.use(
  helmet({
    contentSecurityPolicy: {
      useDefaults: true,
      directives: {
        "script-src": ["'self'", "'unsafe-inline'"],
        "img-src": ["'self'"]
      }
    }
    // contentSecurityPolicy: false
  })
);
app.use(cors(config.cors));
app.use(rateLimit(config.rate));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(morgan("dev"));

morgan.token("rid", (req) => req.id);
app.use(requestId);
app.use(morgan(':method :url :status - rid=:rid - :response-time ms'));

const __dirname = path.dirname(fileURLToPath(import.meta.url));
app.use(express.static(path.join(__dirname, "..", "public")));

// Root index (simple API metadata)
app.get("/", (_req, res) => {
  res.json({
    name: "Week-05 REST API",
    version: "0.1.0",
    endpoints: ["/health", "/users", "/docs", "/articles", "/comments"],
  });
});
app.get("/favicon.ico", (_req, res) => res.status(204).end());
app.get("/health", (_req, res) => res.json({ ok: true }));
app.use("/users", usersRouter);
app.use("/articles", articlesRouter);
app.use("/comments", commentsRouter);

if (process.env.NODE_ENV !== "test") {
  const mod = await import("./docs/openapi.json", { with: {type: "json"}});
  const spec = mod.default;
  app.use("/docs", swaggerUi.serve, swaggerUi.setup(spec));
}

app.use(notFound);
app.use(errorHandler);

export default app;
