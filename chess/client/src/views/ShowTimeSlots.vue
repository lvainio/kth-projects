<template>
  <div class="container">
    <div class="row">
      <div class="col"></div>
      <div class="col">
        <p>Player 1</p>
        <Board />
        <p>Player 2</p>
      </div>
      <div class="col"></div>
    </div>
  </div>
</template>

<script>
import Board from "./Board.vue";

export default {
  name: "Play",
  components: {
    Board,
  },

  data: () => ({
    timeslots: [],
  }),

  async mounted() {
    await fetch("/api/timeslots/data", {
      method: "POST",
    })
      .then((res) => res.json())
      .then(({ timeslots }) => {
        this.timeslots = timeslots;
      })
      .catch(console.error);

    const { socket } = this.$root;
    socket.on("addTimeslot", (timeslot) => {
      this.timeslots = [...this.timeslots, timeslot];
    });
    socket.on("deleteTimeslot", (timeslot) => {
      const index = this.timeslots.findIndex(
        (e) => e.admin === timeslot.admin && e.time === timeslot.time
      );
      if (index >= 0) {
        this.timeslots.splice(index, 1);
      }
    });
    socket.on("updateReserve", (timeslot) => {
      const index = this.timeslots.findIndex(
        (e) => e.admin === timeslot.admin && e.time === timeslot.time
      );

      if (index >= 0) {
        this.timeslots[index].reserved = timeslot.reserved;
      }
    });
    socket.on("bookTimeslot", (timeslot) => {
      const index = this.timeslots.findIndex(
        (e) => e.admin === timeslot.admin && e.time === timeslot.time
      );
      if (index >= 0) {
        this.timeslots[index] = timeslot;
      }
    });
  },

  methods: {
    async reserveTimeslot(admin, time) {
      const { commit } = this.$store;
      const { push } = this.$router;

      commit("setReservedTimeSlot", { admin, time });
      await fetch("/api/timeslots/reserve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ admin, time }),
      })
        .then((res) => res.json())
        .then(({ success }) => {
          if (success) {
            push("/booking");
          } else {
            push("/timeslots");
          }
        })
        .catch(console.error);
    },
  },
};
</script>

<style scoped>
p {
  margin: 0;
}
</style>
