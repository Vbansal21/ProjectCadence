import { Router } from "express";
import { validate } from "../middleware/validate.js";
import { articleCreate, articleUpdate } from "../schemas/articles.schema.js";
import { listArticles, createArticle, getArticle, updateArticle, removeArticle } from "../controllers/articles.controller.js";
import { validateUuidParam } from "../middleware/validateParam.js";
import { apiKey } from "../middleware/apiKey.js";

const router = Router();
router.post("/", apiKey(), validate(articleCreate), createArticle);
router.get("/", listArticles);
router.get("/:id", validateUuidParam("id"), getArticle);
router.put("/:id", apiKey(), validate(articleUpdate), updateArticle);
router.delete("/:id", apiKey(), removeArticle);

export default router;
