<template>
  <div class="board">
    <slot name="popup"></slot>

    <div v-for="(row, rowIndex) in board" :key="rowIndex" class="board-row">
      <div
        v-for="(square, colIndex) in row"
        :key="colIndex"
        class="square"
        :class="getSquareColorClass(rowIndex, colIndex)"
        @click="
          selectPiece(rowIndex, colIndex);
          movePiece(rowIndex, colIndex);
        "
      >
        <Piece
          v-if="square.piece"
          :color="square.piece.color"
          :type="square.piece.type"
        />
      </div>
    </div>
  </div>
</template>

<script>
import Piece from "./Piece.vue";

export default {
  name: "Board",
  components: {
    Piece,
  },

  data: () => ({
    board: null,
    selectedPiece: {
      rowIndex: null,
      colIndex: null,
    },
  }),

  props: {
    playerColor: {
      type: String,
      required: true,
    }
  },

  mounted() {
    this.generateBoard();
    // TODO: probs a websocket to listen to when the other player has made its turn
  },

  methods: {
    /**
     * Generates the initial state of the chess board.
     */
    generateBoard() {
      const types = ["rook", "knight", "bishop", "queen", "king", "bishop", "knight", "rook"];

      this.board = new Array(8).fill(null).map((_, rowIndex) => {
        const color = rowIndex < 2 ? "black" : "white";
        return new Array(8).fill(null).map((_, colIndex) => {
          if (rowIndex === 0 || rowIndex === 7) {
            return { piece: { color, type: types[colIndex] } };
          } else if (rowIndex === 1 || rowIndex === 6) {
            return { piece: { color, type: "pawn" } };
          } else {
            return { piece: null };
          }
        });
      });
    },

    /**
     * Returns the color class of a square.
     * @param {Integer} rowIndex
     * @param {Integer} colIndex
     */
    getSquareColorClass(rowIndex, colIndex) {
      const isWhiteSquare = (rowIndex + colIndex) % 2 === 0;
      return isWhiteSquare ? "white" : "black";
    },

    /**
     * If a square with the players piece is clicked then that piece is selected to be moved on the next click.
     * @param {Integer} rowIndex
     * @param {Integer} colIndex
     */
    selectPiece(rowIndex, colIndex) {
      const { piece } = this.board[rowIndex][colIndex];
      if (piece !== null && piece.color === this.playerColor) {
        this.selectedPiece.rowIndex = rowIndex;
        this.selectedPiece.colIndex = colIndex;
      }
    },

    /**
     * If a piece is already selected and a valid square is clicked then the piece moves.
     * @param {Integer} rowIndex
     * @param {Integer} colIndex
     */
    movePiece(rowIndex, colIndex) {
      const { piece } = this.board[rowIndex][colIndex];
      if (piece === null && this.selectedPiece.rowIndex !== null) {
        this.board[rowIndex][colIndex].piece = this.board[this.selectedPiece.rowIndex][this.selectedPiece.colIndex].piece;
        this.board[this.selectedPiece.rowIndex][this.selectedPiece.colIndex].piece = null;
      }

      // TODO: insert a million checks here and on the server side to see if move is valid.
      // TODO: communicate to the server via fetch or socket what move is to be made
      // TODO: only move allowed when its ur turn

    }
  }
};
</script>

<style scoped>
.board {
  position: relative;
  border: 20px solid #9a5e22;
  border-radius: 10px;
  width: fit-content;
}

.board-row {
  display: flex;
  margin: 0;
  padding: 0;
}

.square {
  background-size: contain;
  background-position: center center;
  background-repeat: no-repeat;
  height: 70px;
  width: 70px;
}

.square.white {
  background-color: #ffe9c5;
}

.square.black {
  background-color: #d58d45;
}
</style>
