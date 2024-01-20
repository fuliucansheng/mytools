import { recoTheme } from "vuepress-theme-reco";

const navbar = [
  { text: "AI实验室", link: "/ailab", icon: "Idea" },
  {
    text: "教程",
    icon: "SubVolume",
    children: [
      {
        text: "unitorch框架",
        children: [
          { text: "文档指南", link: "/unitorch" },
          { text: "快速开始", link: "/unitorch/configuration" },
        ],
      },
      {
        text: "mytools工具箱",
        children: [
          { text: "文档指南", link: "/mytools" },
          { text: "工具列表", link: "/mytools/spaces" },
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
          { text: "基础知识", link: "/blogs/nlp/base" },
          { text: "NLU系列", link: "/blogs/nlp/nlu" },
          { text: "NLG系列", link: "/blogs/nlp/nlg" },
          { text: "LLM系列", link: "/blogs/nlp/llm" },
        ],
      },
      {
        text: "计算机视觉",
        children: [
          { text: "基础知识", link: "/blogs/cv/base" },
          { text: "分类系列", link: "/blogs/cv/base" },
          { text: "检测系列", link: "/blogs/cv/base" },
          { text: "分割系列", link: "/blogs/cv/base" },
          { text: "Diffusion系列", link: "/blogs/cv/base" },
        ],
      },
      {
        text: "多模态",
        children: [
          { text: "基础知识", link: "/blogs/mm/base" },
          { text: "CLIP系列", link: "/blogs/mm/clip" },
          { text: "MLLM系列", link: "/blogs/mm/mllm" },
        ],
      },
    ],
  },
  { text: "常用工具", link: "/tools", icon: "ToolBox" },
  { text: "关于我", link: "/about", icon: "Information" },
];

export default recoTheme({
  logo: "/favicon2.png",
  hostname: "https://fuliucansheng.vercel.app/",
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
