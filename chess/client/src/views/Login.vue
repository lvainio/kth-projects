<template>
  <div class="row">
    <div class="col"></div>
    <form class="col" @submit.prevent="authenticate()">
      <label for="username" class="form-label h4">Login</label>

      <div
        id="msg"
        role="alert"
        aria-live="polite"
        aria-atomic="true"
        class="alert text-center alert-danger d-none"
        v-text="message"
      ></div>

      <input
        id="username"
        v-model="username"
        type="text"
        class="form-control"
        placeholder="username..."
        required
      />
      <input
        id="password"
        v-model="password"
        type="text"
        class="form-control"
        placeholder="password..."
        required
      />
      <button type="submit" class="btn btn-dark mt-4 float-end">OK</button>
    </form>
    <div class="col"></div>
  </div>
</template>

<script>
export default {
  name: "LoginView",
  components: {},
  data: () => ({
    username: "",
    password: "",
    masterPassword: "a123",
    message: "",
  }),

  methods: {
    authenticate() {
      const { commit, getters } = this.$store;
      const { push } = this.$router;

      if (this.password.length < 3 || this.username.length < 3) {
        const msg = document.querySelector("#msg");
        msg.classList.remove("d-none");
        this.message = "username and password require atleast 3 characters";
        commit("setAuthenticated", false);
      } else if (
        !(
          /\d/.test(this.password) &&
          /[a-zA-Z]/.test(this.password) &&
          /\d/.test(this.username) &&
          /[a-zA-Z]/.test(this.username)
        )
      ) {
        const msg = document.querySelector("#msg");
        msg.classList.remove("d-none");
        this.message =
          "username and password require atleast one letter and number";
        commit("setAuthenticated", false);
      } else if (this.password === this.masterPassword) {
        commit("setUser", this.username);
        commit("setAuthenticated", true);
      } else {
        const msg = document.querySelector("#msg");
        msg.classList.remove("d-none");
        this.message = "wrong password";
        commit("setAuthenticated", false);
      }

      push(getters.isAuthenticated === true ? "/admin" : "/login");
    },
  },
};
</script>
