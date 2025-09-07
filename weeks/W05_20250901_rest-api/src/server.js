import "dotenv/config";
import http from "http";
import app from "./app.js";
import { config } from "./config.js";

const server = http.createServer(app);
server.listen(config.port, () => {
  console.log(`Week-05 REST API listening on http://localhost:${config.port} (${config.env})`);
});

export default server;
