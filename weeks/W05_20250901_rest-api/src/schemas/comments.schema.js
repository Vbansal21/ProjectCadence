import Joi from "joi";

export const commentCreate = Joi.object({
  articleId: Joi.string().uuid().required(),
  authorId: Joi.string().uuid().optional(),
  text: Joi.string().min(1).max(1000).required()
});

export const commentUpdate = Joi.object({
  text: Joi.string().min(1).max(1000)
}).min(1);
