import { createRouter, createWebHistory } from "vue-router";
import store from "../store";

import Login from "../views/Login.vue";
import Admin from "../views/Admin.vue";
import ShowTimeSlots from "../views/ShowTimeSlots.vue";
import Booking from "../views/Booking.vue";
import Register from "../views/Register.vue";
import Play from "../views/Play.vue";
import Lobby from "../views/Lobby.vue";

const routes = [
  {
    path: "/",
    redirect: "/login", // TODO: login or admin?
  },
  {
    path: "/play",
    component: Play,
  },
  {
    path: "/login",
    component: Login,
  },
  {
    path: "/admin",
    component: Admin,
  },
  {
    path: "/lobby",
    component: Lobby,
  },
  {
    path: "/timeslots",
    component: ShowTimeSlots,
  },
  {
    path: "/booking",
    component: Booking,
  },
  {
    path: "/register",
    component: Register,
  },
];

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes,
});

// Authentication guard.
router.beforeEach((to, from, next) => {
  if (
    store.state.authenticated ||
    to.path === "/login" ||
    to.path === "/timeslots" ||
    to.path === "/booking" ||
    to.path === "/register" ||
    to.path === "/play" ||
    to.path === "/lobby"
  ) {
    next();
  } else {
    console.info("Unauthenticated user. Redirecting to login page.");
    next("/login");
  }
});

export default router;
