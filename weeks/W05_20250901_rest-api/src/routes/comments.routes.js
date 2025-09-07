import { Router } from "express";
import { validate } from "../middleware/validate.js";
import { commentCreate, commentUpdate } from "../schemas/comments.schema.js";
import { listComments, createComment, getComment, updateComment, removeComment } from "../controllers/comments.controller.js";
import { validateUuidParam } from "../middleware/validateParam.js";

const router = Router();
router.get("/", listComments);
router.post("/", validate(commentCreate), createComment);
router.get("/:id", validateUuidParam("id"), getComment);
router.put("/:id", validate(commentUpdate), updateComment);
router.delete("/:id", removeComment);

export default router;
