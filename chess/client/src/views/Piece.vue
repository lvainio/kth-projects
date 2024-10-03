<template>
  <div class="piece" :class="[color, type]">
    <img :src="imagePath" :alt="type" />
  </div>
</template>

<script>
export default {
  name: "Piece",
  props: {
    color: {
      type: String,
      required: true,
    },
    type: {
      type: String,
      required: true,
    },
  },
  data: () => ({
    imagePath: "",
  }),
  mounted() {
    this.imagePath = `/${this.color}-${this.type}.svg`;
  },
};
</script>

<style scoped>
.piece {
  width: 100%;
  height: 100%;
  background-repeat: no-repeat;
  background-position: center;
}

img {
  width: 100%;
  height: 100%;
}
</style>
