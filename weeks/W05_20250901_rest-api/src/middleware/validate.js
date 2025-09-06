export const validate = (schema) => (req, _res, next) => {
    const { error, value } = schema.validate(req.body, {
      abortEarly: false, stripUnknown: true
    });
    if (error) return next(error);
    req.validated = value;
    next();
  };
