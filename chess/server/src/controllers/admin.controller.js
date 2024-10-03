import { Router } from "express";
import db from "../database/db.js";
import model from "../model.js";

const router = Router();

/**
 * requireAuth is a middleware function that limit access to an endpoint to authenticated users.
 * @param {Request} req
 * @param {Response} res
 * @param {Function} next
 * @returns {void}
 */
const requireAuth = (req, res, next) => {
  if (
    req.session.username === undefined &&
    req.path !== "/login" &&
    req.path !== "/users/me" &&
    req.path !== "/register"
  ) {
    res.status(401).end();
    return;
  }
  next();
};

router.get("/users/me", (req, res) => {
  if (req.session.username !== undefined) {
    res.status(200).json({ authenticated: true });
  } else {
    res.status(200).json({ authenticated: false });
  }
});

router.post("/admin/logout", (req, res) => {
  req.session.destroy();

  res.status(200).json({});
});

router.post("/admin/add", async (req, res) => {
  const { time } = req.body;
  const { username } = req.session;

  if (username !== undefined) {
    console.log(`Inserted time for user ${req.session.username}`);

    model.addtime(username, time);
    res.status(200).json({ success: true });
  } else {
    res.status(401).json({ success: false });
  }
});

router.post("/admin/delete", async (req, res) => {
  const { time } = req.body;
  const { username } = req.session;

  if (username !== undefined) {
    console.log(`Deleted time for user ${req.session.username}`);
    model.deletetime(username, time);
    res.status(200).json({ success: true });
  } else {
    res.status(401).json({ success: false });
  }
});

router.post("/admin/data", async (req, res) => {
  const { username } = req.session;
  const timeslots = [];
  if (username !== undefined) {
  }
  res.status(200).json({ admin: username, timeslots });
});

router.post("/login", async (req, res) => {
  const { username, password } = req.body;

  if (user === undefined || user.password !== password) {
    res
      .status(401)
      .json({ authenticated: false, message: "Wrong username or password" });
  } else {
    req.session.username = username;
    res.status(200).json({ authenticated: true });
  }
});

router.post("/register", async (req, res) => {
  const { username, password } = req.body;

  if (password.length < 3 || username.length < 3) {
    res.status(401).json({
      success: false,
      message: "Username and password require atleast 3 characters",
    });
  } else if (
    !(
      /\d/.test(password) &&
      /[a-zA-Z]/.test(password) &&
      /\d/.test(username) &&
      /[a-zA-Z]/.test(username)
    )
  ) {
    res.status(401).json({
      success: false,
      message:
        "Username and password require atleast one letter and one number",
    });
  } else {
    const user = await db.get("SELECT * FROM user WHERE username =?", [
      username,
    ]);
    if (user === undefined) {
      console.log(`User ${username} inserted`);
      res.status(200).json({ success: true });
    } else {
      res
        .status(401)
        .json({ success: false, message: "Username already exist" });
    }
  }
});

export default { router, requireAuth };
