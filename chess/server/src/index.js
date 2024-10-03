import betterLogging from "better-logging";
import express from "express";
import expressSession from "express-session";
import socketIOSession from "express-socket.io-session";
import { createServer } from "http";
import { Server } from "socket.io";
import resolvePath from "./util.js";
import model from "./model.js";
import admin from "./controllers/admin.controller.js";
import timeslot from "./controllers/timeslot.controller.js";

const port = 8989;
const app = express();
const server = createServer(app);
const io = new Server(server);

const { Theme } = betterLogging;
betterLogging(console, {
  color: Theme.green,
});

// Enable debug output
console.logLevel = 4;

// Register a custom middleware for logging incoming requests
app.use(
  betterLogging.expressMiddleware(console, {
    ip: { show: true, color: Theme.green.base },
    method: { show: true, color: Theme.green.base },
    header: { show: false },
    path: { show: true },
    body: { show: true },
  })
);

// Configure session management
const sessionConf = expressSession({
  secret: "Super secret! Shh! Do not tell anyone...",
  resave: true,
  saveUninitialized: true,
});
app.use(sessionConf);
io.use(
  socketIOSession(sessionConf, {
    autoSave: true,
    saveUninitialized: true,
  })
);

// Serve static files
app.use(express.static(resolvePath("client", "dist")));

// Register middlewares that parse the body of the request, available under req.body property
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Bind REST controllers to /api/*
app.use("/api", timeslot.router);
app.use("/api", admin.requireAuth, admin.router);

// Redirect any path that isn't handled to index.html
app.get("/*", (req, res) => {
  res.redirect("/index.html");
});

// Initialize a model
model.init(io);

// Handle socket.io connections
io.on("connection", (socket) => {
  const { session } = socket.handshake;
  session.socketID = socket.id;
  session.save((err) => {
    if (err) console.error(err);
    else console.debug(`Saved socketID: ${session.socketID}`);
  });
});

server.listen(port, () => {
  console.log(`Listening on http://localhost:${port}/`);
});
