<template>
  <div class="row">
    <div class="col"></div>
    <div class="col">
      <h1>Signed in as {{ admin }}</h1>

      <form id="my-form" @submit.prevent="addTime">
        <label for="admin" class="form-label h4"></label>
        <input id="time-input" v-model="time" type="time" required />
        <button id="submit-button" type="submit" class="btn btn-dark">
          add time
        </button>
      </form>

      <div id="time-list" class="col list-group">
        <div
          v-for="timeslot in timeslots"
          :key="timeslot.time"
          class="list-group-item list-group-item-action my-2 py-2"
        >
          <p>{{ timeslot.time }}</p>
          <button type="button" class="btn btn-dark" @click="deleteTime">
            Delete
          </button>
          <p v-if="timeslot.student !== null" class="student">
            Booked by: {{ timeslot.student }}
          </p>
        </div>
      </div>
    </div>
    <div class="col"></div>
  </div>
</template>

<script>
export default {
  name: "AdminView",
  components: {},

  data: () => ({
    timeslots: [],
    time: "",
    admin: "",
  }),

  mounted() {
    const { socket } = this.$root;
    socket.on("bookTimeslot", (timeslot) => {
      if (timeslot.admin !== this.admin) {
        return;
      }
      const index = this.timeslots.findIndex((e) => e.time === timeslot.time);
      if (index >= 0) {
        this.timeslots[index].student = timeslot.student;
      }
    });
  },

  created() {
    fetch("/api/admin/data", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    })
      .then((res) => res.json())
      .then(({ admin, timeslots }) => {
        this.admin = admin;
        this.timeslots = timeslots;
      })
      .catch(console.error);
  },

  methods: {
    deleteTime(event) {
      const parentDiv = event.target.parentElement;
      const time = parentDiv.children[0].innerHTML;

      fetch("/api/admin/delete", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ time }),
      })
        .then((res) => res.json())
        .then(({ success }) => {
          if (success) {
            for (let index = 0; index < this.timeslots.length; index += 1) {
              const timeslot = this.timeslots[index];
              if (timeslot.time === time) {
                this.timeslots.splice(index, 1);
                break;
              }
            }
          }
        })
        .catch(console.error);
    },

    addTime() {
      fetch("/api/admin/add", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ time: this.time }),
      })
        .then((res) => res.json())
        .then(({ success }) => {
          if (success) {
            this.timeslots.push({
              time: this.time,
              student: null,
            });
          }
          this.time = "";
        })
        .catch(console.error);
    },
  },
};
</script>

<style></style>
