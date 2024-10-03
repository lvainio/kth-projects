<template>
  <nav class="navbar navbar-expand-md navbar-dark bg-dark">
    <button
      class="navbar-toggler mx-2 mb-2"
      type="button"
      data-bs-toggle="collapse"
      data-bs-target="#navbarNav"
    >
      <span class="navbar-toggler-icon"></span>
    </button>
    <div id="navbarNav" class="collapse navbar-collapse mx-2">
      <ul class="navbar-nav">
        <li v-if="!isAuthenticated()" class="nav-item">
          <a class="nav-link" href="#" @click="redirect('/login')">Login</a>
        </li>
        <li v-if="isAuthenticated()" class="nav-item">
          <a class="nav-link" href="#" @click="redirect('/admin')">Admin</a>
        </li>
        <li class="nav-item">
          <a class="nav-link" href="#" @click="redirect('/play')">Play</a>
        </li>
        <li v-if="!isAuthenticated()" class="nav-item">
          <a class="nav-link" href="#" @click="redirect('/register')"
            >Register</a
          >
        </li>
        <li v-if="isAuthenticated()" class="nav-item">
          <a class="nav-link" href="#" @click="logout()">Logout</a>
        </li>
      </ul>
    </div>
  </nav>
  <section class="container-fluid py-4">
    <router-view />
  </section>
</template>

<script>
import "bootstrap";
import io from "socket.io-client";

export default {
  // TODO: ShowTimeSlots, Booking, Socket stuff, Styling, Lint, Testing, Show which student booked, Logout,
  // Fix navigation bar depending on authorization, 10 second timer, time chosen by other student should get occupied,
  // We are supposed to use model for assistand and timeslot, Why model when we have database?, timeout both on serverside
  // and client side. remove session after logout req.session.destroy. Maybe use ID in timeslots? socket msg to adminview if
  // one of their times get booked (maybe make this as room in chat application?). maybe reqauth fix by splitting up into two routers
  // in admin controller?

  name: "App",
  components: {},
  data: () => ({
    socket: io(/* socket.io options */).connect(),
  }),
  created() {
    const { commit } = this.$store;
    const { push } = this.$router;

    fetch("/api/users/me")
      .then((res) => res.json())
      .then(({ authenticated }) => {
        commit("setAuthenticated", authenticated);
        push(authenticated === true ? "/admin" : "/login");
      })
      .catch(console.error);
  },
  methods: {
    redirect(target) {
      this.$router.push(target);
    },
    logout() {
      this.$store.commit("setAuthenticated", false);

      fetch("/api/admin/logout", {
        method: "POST",
      })
        .then((res) => res.json())
        .then(() => {})
        .catch(console.error);

      this.redirect("/login");
    },
    isAuthenticated() {
      return this.$store.getters.isAuthenticated;
    },
  },
};
</script>

<style>
@import url("bootstrap/dist/css/bootstrap.css");

html,
body {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
  background-color: #ffe9c5;
}
</style>
