import { navbar } from "vuepress-theme-hope";

export const NavbarConfig = navbar([
  {
    text: "首页",
    icon: "home",
    link: "/",
  },
  {
    text: "AI实验室",
    icon: "lightbulb",
    link: "/ailab",
  },
  {
    text: "Web技术",
    icon: "server",
    link: "/web",
  },
  {
    text: "大数据技术",
    icon: "database",
    link: "/data",
  },
  {
    text: "随笔",
    icon: "book-open",
    link: "/notes",
  },
  {
    text: "工具箱",
    icon: "toolbox",
    link: "/tools",
  },
  {
    text: "关于",
    icon: "circle-exclamation",
    link: "/about",
  },
]);
