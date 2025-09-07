import Joi from "joi";

export const articleCreate = Joi.object({
  title: Joi.string().min(3).max(120).required(),
  body: Joi.string().min(1).required(),
  authorId: Joi.string().uuid().required()
});

export const articleUpdate = Joi.object({
  title: Joi.string().min(3).max(120),
  body: Joi.string().min(1)
}).min(1);
