<template>
  <div class="row">
    <div class="col"></div>
    <div class="col">
      <h4>Book time</h4>

      <p>TA: {{ timeslot.admin }}</p>
      <p>Time: {{ timeslot.time }}</p>

      <form id="my-form" @submit.prevent="bookTime()">
        <label for="username" class="form-label h4"></label>
        <input
          id="student-input"
          v-model="studentName"
          type="text"
          placeholder="name"
          required
        />
        <button id="book-button" type="submit" class="btn btn-dark">
          Book time
        </button>
        <button
          id="go-back-button"
          type="button"
          class="btn btn-dark"
          @click="cancelReservation()"
        >
          Go back
        </button>
      </form>

      <h5>Time left: {{ timerCount }}</h5>
    </div>
    <div class="col"></div>
  </div>
</template>

<script>
export default {
  name: "BookingView",
  components: {},
  data: () => ({
    timeslot: {},
    timerCount: 10,
    studentName: "",
  }),

  watch: {
    timerCount: {
      handler(value) {
        if (value > 0) {
          setTimeout(() => {
            this.timerCount -= 1;
          }, 1000);
        } else {
          this.redirect("/timeslots");
        }
      },
      immediate: true,
    },
  },

  created() {
    this.timeslot = this.$store.getters.getReservedTimeSlot;
  },

  methods: {
    redirect(target) {
      this.$router.push(target);
    },
    async bookTime() {
      await fetch("/api/booktime", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          time: this.timeslot.time,
          admin: this.timeslot.admin,
          student: this.studentName,
        }),
      })
        .then((res) => res.json())
        .then(() => {
          const { push } = this.$router;
          push("/timeslots");
        })
        .catch(console.error);
    },
    async cancelReservation() {
      await fetch("/api/cancelreservation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          timeslot: this.timeslot,
        }),
      })
        .then((res) => res.json())
        .then(() => {
          const { push } = this.$router;
          push("/timeslots");
        })
        .catch(console.error);
    },
  },
};
</script>

<style></style>
