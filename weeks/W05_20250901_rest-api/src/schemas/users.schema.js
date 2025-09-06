import Joi from "joi";

export const userCreate = Joi.object({
  name: Joi.string().min(2).max(64).required(),
  email: Joi.string().email().required()
});

export const userUpdate = Joi.object({
  name: Joi.string().min(2).max(64),
  email: Joi.string().email()
}).min(1);
