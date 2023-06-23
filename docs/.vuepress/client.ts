import { defineClientConfig } from "@vuepress/client";
import { onMounted } from "vue";

export default defineClientConfig({
  setup() {
    onMounted(() => {
      console.log("Welcome to 拂柳残声の技术笔记。");
    });
  },
});
