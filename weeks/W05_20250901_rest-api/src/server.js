import http from "http";
import app from "./app.js";

const port = process.env.PORT || 3000;
http.createServer(app).listen(port, () => {
  console.log(`Week-05 REST API listening on http://localhost:${port}`);
});
