<template>
  <div class="container">
    <div class="row">
      <div class="col"></div>
      <div class="col">
        <h4>Player black name</h4>
        <!-- TODO: replace with player name -->
        <!-- TODO: replace with players color -->
        <Board :player-color="'black'">
          <template #popup>
            <Popup v-if="gameOver">
              <h1>White won!</h1>
              <!-- TODO: replace with whomever won -->
              <button class="btn btn-dark" @click="redirect('/lobby')">
                Back to lobby
              </button>
            </Popup>
          </template>
        </Board>
        <h4>Player white name</h4>
        <!-- TODO: replace with player name -->
      </div>
      <div class="col"></div>
    </div>
  </div>
</template>

<script>
import Board from "./Board.vue";
import Popup from "./Popup.vue";

export default {
  name: "Play",
  components: {
    Board,
    Popup,
  },

  // implement websocket thing to check for wins.
  // TODO: we need websocket rooms for each game. 

  data: () => ({
    whitePlayer: "",
    blackPlayer: "",
    gameOver: false, // TODO: set this or some other boolean when the game is over
  }),

  mounted() {
    // TODO: fetch the users names from server
    // TODO: websocket stuff to notify that a player has won or the game drew or similar.
    // TODO: then change gameOver to true so that popup appears.
    // TODO: save into players database the game.
  },

  methods: {
    redirect(target) {
      this.$router.push(target);
    },
  },
};
</script>

<style scoped>
h4 {
  text-align: center;
  margin: 5px;
}
</style>
