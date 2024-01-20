import { defineUserConfig } from "vuepress";
import recoTheme from "./theme.js";

export default defineUserConfig({
  lang: "zh-CN",
  title: "拂柳残声の技术主站",
  description: "专注于互联网技术与人工智能",
  theme: recoTheme,
  head: [
    ['link', { rel: 'icon', type: "image/x-icon", href: `./favicon1.png` }]
 ]
});
