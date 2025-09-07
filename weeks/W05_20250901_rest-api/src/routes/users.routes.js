import { Router } from "express";
import { listUsers, createUser, getUser, updateUser, removeUser } from "../controllers/users.controller.js";
import { validate } from "../middleware/validate.js";
import { userCreate, userUpdate } from "../schemas/users.schema.js";
import { validateUuidParam } from "../middleware/validateParam.js";

const router = Router();

router.get("/", listUsers);
router.post("/", validate(userCreate), createUser);
router.get("/:id", validateUuidParam(), getUser);
router.put("/:id", validateUuidParam(), validate(userUpdate), updateUser);
router.delete("/:id", validateUuidParam(), removeUser);

export default router;
