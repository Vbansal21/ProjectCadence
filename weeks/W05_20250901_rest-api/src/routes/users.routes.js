import { Router } from "express";
import { listUsers, createUser, getUser, updateUser, removeUser } from "../controllers/users.controller.js";
import { validate } from "../middleware/validate.js";
import { userCreate, userUpdate } from "../schemas/users.schema.js";

const router = Router();

router.get("/", listUsers);
router.post("/", validate(userCreate), createUser);
router.get("/:id", getUser);
router.put("/:id", validate(userUpdate), updateUser);
router.delete("/:id", removeUser);

export default router;
