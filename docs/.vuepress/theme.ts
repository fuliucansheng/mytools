import { recoTheme } from "vuepress-theme-reco";

const navbar = [
  { text: "AI实验室", link: "/", icon: "Idea" },
  {
    text: "教程",
    icon: "SubVolume",
    children: [
      {
        text: "unitorch",
        children: [
          { text: "主题配置", link: "/docs/theme/frontmatter" },
          { text: "Markdown 扩展", link: "/docs/theme/custom-container" },
          { text: "高级", link: "/docs/theme/custom-catalog-title" },
        ],
      },
      {
        text: "mytools",
        children: [
          { text: "page", link: "/docs/plugins/page" },
          { text: "comments", link: "/docs/plugins/comments" },
          { text: "vue-previews", link: "/docs/plugins/vue-previews" },
          { text: "bulletin-popover", link: "/docs/plugins/bulletin-popover" },
        ],
      },
    ],
  },
  {
    text: "博客",
    icon: "Blog",
    children: [
      {
        text: "自然语言处理",
        children: [
          { text: "page", link: "/docs/plugins/page" },
          { text: "comments", link: "/docs/plugins/comments" },
          { text: "vue-previews", link: "/docs/plugins/vue-previews" },
          { text: "bulletin-popover", link: "/docs/plugins/bulletin-popover" },
        ],
      },
      {
        text: "计算机视觉",
        children: [
          { text: "page", link: "/docs/plugins/page" },
          { text: "comments", link: "/docs/plugins/comments" },
          { text: "vue-previews", link: "/docs/plugins/vue-previews" },
          { text: "bulletin-popover", link: "/docs/plugins/bulletin-popover" },
        ],
      },
      {
        text: "多模态",
        children: [
          { text: "page", link: "/docs/plugins/page" },
          { text: "comments", link: "/docs/plugins/comments" },
          { text: "vue-previews", link: "/docs/plugins/vue-previews" },
          { text: "bulletin-popover", link: "/docs/plugins/bulletin-popover" },
        ],
      },
      {
        text: "推荐系统",
        children: [
          { text: "page", link: "/docs/plugins/page" },
          { text: "comments", link: "/docs/plugins/comments" },
          { text: "vue-previews", link: "/docs/plugins/vue-previews" },
          { text: "bulletin-popover", link: "/docs/plugins/bulletin-popover" },
        ],
      },
      {
        text: "强化学习",
        children: [
          { text: "page", link: "/docs/plugins/page" },
          { text: "comments", link: "/docs/plugins/comments" },
          { text: "vue-previews", link: "/docs/plugins/vue-previews" },
          { text: "bulletin-popover", link: "/docs/plugins/bulletin-popover" },
        ],
      },
    ],
  },
  { text: "关于我", link: "/about", icon: "Information" },
];

export default recoTheme({
  logo: "/favicon2.png",
  hostname: "https://mytools.fuliucansheng.vercel.app/",
  colorMode: "light",
  autoSetBlogCategories: false,
  autoAddCategoryToNavbar: false,
  autoSetSeries: false,
  catalogTitle: "目录",
  navbar: navbar,
  commentConfig: {
    type: "giscus",
    options: {
      repo: "fuliucansheng/mytools",
      repoId: "R_kgDOJyeEew",
      category: "General",
      categoryId: "DIC_kwDOJyeEe84CXasP",
    },
  },
});
