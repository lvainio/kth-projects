import { Router } from "express";
import model from "../model.js";
import db from "../database/db.js";

const router = Router();

router.post("/timeslots/data", async (req, res) => {
  res.status(200).json({ timeslots: model.getTimeslots() });
});

router.post("/timeslots/reserve", async (req, res) => {
  const { admin, time } = req.body;
  const timeslot = model.getTimeslot(admin, time);
  if (
    timeslot === undefined ||
    timeslot.student !== null ||
    timeslot.reserved
  ) {
    res.status(403).json({ success: false });
    return;
  }
  res.status(200).json({ success: true });
  timeslot.reserved = true;
  model.updateReserve(timeslot);
  setTimeout(() => {
    if (timeslot.student === null || timeslot.reserved) {
      timeslot.reserved = false;
      model.updateReserve(timeslot);
    }
  }, 10000);
});

router.post("/booktime", async (req, res) => {
  const { time, admin, student } = req.body;

  const timeslot = model.getTimeslot(admin, time);

  if (timeslot === undefined || timeslot.student !== null) {
    res.status(403).json({ success: false });
    return;
  }

  statement.run(student, timeslot.id);
  statement.finalize();

  timeslot.student = student;
  timeslot.reserved = false;
  model.bookTime(timeslot);

  res.status(200).json({ success: true });
});

router.post("/cancelreservation", async (req, res) => {
  const { timeslot } = req.body;

  model.cancelReservation(timeslot);

  res.status(200).json({});
});

export default { router };
