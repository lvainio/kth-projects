/**
 * @class Timeslot
 */
class Timeslot {
  constructor(id, admin, student, time) {
    this.id = id;
    this.admin = admin;
    this.student = student;
    this.time = time;
    this.reserved = false;
  }
}

export default Timeslot;
