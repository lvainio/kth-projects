<template>
  <div class="container">
    <div class="popup">
      <slot />
    </div>
  </div>
</template>

<style scoped>
.container {
  position: absolute;
  top: 0;
  left: 0;
  z-index: 100;
  background-color: rgb(0 0 0 / 30%);
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
}

.popup {
  background: #ffe9c5;
  padding: 32px;
  border-radius: 10px;
  text-align: center;
  width: 60%;
}
</style>
