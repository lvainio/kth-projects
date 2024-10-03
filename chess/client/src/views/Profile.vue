<template>
  <div class="row">
    <div class="col"></div>
    <div class="col">
      <h1>Signed in as {{ username }}</h1>

      <form id="my-form" @submit.prevent="addTime">
        <label for="username" class="form-label h4"></label>

        <input id="time-input" v-model="time" type="time" required />
        <button type="submit" class="btn btn-dark">add time</button>
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
          <p v-if="timeslot.isBooked" align="right">
            Booked by: {{ timeslot.studentName }}
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
    username: "",
  }),

  mounted() {
    this.username = this.$store.getters.getUsername;
    this.timeslots = this.$store.getters.getTimeSlotsByName;
  },

  methods: {
    deleteTime(event) {
      const { commit, getters } = this.$store;
      const parentDiv = event.target.parentElement;

      const time = parentDiv.children[0].innerHTML;
      parentDiv.remove();
      commit("deleteTime", [time, getters.getUsername]);
    },

    addTime() {
      const { commit, getters } = this.$store;
      this.timeslots.push({
        time: this.time,
        name: getters.getUsername,
        isBooked: false,
      });
      commit("addTime", this.time);
      this.time = "";
    },
  },
};
</script>
