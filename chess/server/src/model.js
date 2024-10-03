import Admin from "./models/admin.model.js";
import Timeslot from "./models/timeslot.model.js";

import db from "./database/db.js";

class Model {
  constructor() {
    this.timeslots = [];
    this.admins = [];

    this.io = undefined;
  }

  /**
   * Initialize the model after its creation.
   * @param {SocketIO.Server} io - The socket.io server instance.
   * @returns {void}
   */
  async init(io) {
    this.io = io;
  }

  /**
   * Add a new time.
   * @param {String} admin
   * @param {Time} time
   */
  async addtime(admin, time) {
    const timeslot = new Timeslot(id, admin, null, time);
    this.timeslots.push(timeslot);
    this.io.emit("addTimeslot", timeslot);
  }

  /**
   * Delete a time.
   * @param {String} admin
   * @param {Time} time
   */
  deletetime(admin, time) {
    const index = this.timeslots.findIndex(
      (e) => e.admin === admin && e.time === time
    );
    if (index >= 0) {
      this.timeslots.splice(index, 1);
      this.io.emit("deleteTimeslot", {
        admin,
        time,
      });
    }
  }

  /**
   * Reserve a time.
   * @param {String} admin
   * @param {Time} time
   */
  updateReserve(timeslot) {
    this.io.emit("updateReserve", timeslot);
  }

  /**
   *
   * @param {*} admin
   * @param {*} time
   * @returns
   */
  getTimeslot(admin, time) {
    const index = this.timeslots.findIndex(
      (e) => e.admin === admin && e.time === time
    );
    if (index >= 0) {
      return this.timeslots[index];
    }
    return undefined;
  }

  cancelReservation(timeslot) {
    const index = this.timeslots.findIndex(
      (e) => e.admin === timeslot.admin && e.time === timeslot.time
    );
    if (index >= 0) {
      this.timeslots[index].reserved = false;
      this.io.emit("updateReserve", this.timeslots[index]);
    }
  }

  bookTime(timeslot) {
    this.io.emit("bookTimeslot", timeslot);
  }

  getTimeslots() {
    return this.timeslots;
  }
}

export default new Model();
