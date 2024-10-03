import { createStore } from "vuex";

export default createStore({
  state: {
    authenticated: false,
    reservedTimeSlot: {},
  },
  getters: {
    isAuthenticated(state) {
      return state.authenticated;
    },
    getReservedTimeSlot(state) {
      return state.reservedTimeSlot;
    },
  },
  mutations: {
    setAuthenticated(state, authenticated) {
      state.authenticated = authenticated;
    },
    setReservedTimeSlot(state, timeslot) {
      state.reservedTimeSlot = timeslot;
    },
  },
  actions: {},
  modules: {},
});
