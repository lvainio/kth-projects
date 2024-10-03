<template>
  <div class="row">
    <div class="col"></div>
    <form class="col" @submit.prevent="register()">
      <label for="username" class="form-label h4">Register</label>
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
      <input
        id="confirm"
        v-model="confirm"
        type="text"
        class="form-control"
        placeholder="confirm password..."
        required
      />
      <button type="submit" class="btn btn-dark mt-4 float-end">
        Register
      </button>
    </form>
    <div class="col"></div>
  </div>
</template>

<script>
export default {
  name: "RegisterView",
  components: {},
  data: () => ({
    username: "",
    password: "",
    confirm: "",
    message: "",
  }),
  methods: {
    register() {
      const { push } = this.$router;
      if (this.password.length < 3 || this.username.length < 3) {
        const msg = document.querySelector("#msg");
        msg.classList.remove("d-none");
        this.message = "Username and password require atleast 3 characters";
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
          "Username and password require atleast one letter and number";
      } else if (this.password !== this.confirm) {
        const msg = document.querySelector("#msg");
        msg.classList.remove("d-none");
        this.message = "Passwords does not match";
      } else {
        fetch("/api/register", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            username: this.username,
            password: this.password,
          }),
        })
          .then((res) => res.json())
          .then(({ success, message }) => {
            if (!success) {
              const msg = document.querySelector("#msg");
              msg.classList.remove("d-none");
              this.message = message;
            }
            push(success === true ? "/login" : "/register");
          })
          .catch(console.error);
      }
    },
  },
};
</script>

<style></style>
