import { defineUserConfig } from "vuepress";
import { hopeTheme, mdEnhance } from "vuepress-theme-hope";
import { shikiPlugin } from "@vuepress/plugin-shiki";
import { searchProPlugin } from "vuepress-plugin-search-pro";

export default defineUserConfig({
  lang: "zh-CN",
  title: "风中追寻の技术人生",
  description: "专注于互联网技术与人工智能",
  theme: hopeTheme({
    logo: "/favicon2.png",
    hostname: "https://fuliucansheng.github.io/",
    iconAssets: [
      "https://cdn.bootcdn.net/ajax/libs/font-awesome/6.4.0/css/all.min.css",
    ],
    iconPrefix: "fas fa-",
    sidebarIcon: true,
    sidebar: {
      "/ailab/": "structure",
      "/web/": "structure",
      "/data/": "structure",
      "/notes/": "structure",
      "/tools/": "structure",
      "/": [],
    },
    navbarIcon: true,
    navbarLayout: {
      start: ["Brand"],
      center: ["Links"],
      end: ["Language", "Repo", "Outlook", "Search"],
    },
    navbar: [
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
        text: "Web技术栈",
        icon: "server",
        link: "/web",
      },
      {
        text: "大数据技术栈",
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
        icon: "info",
        link: "/about",
      },
    ],
    repo: "fuliucansheng",
    repoLabel: "GitHub",
    repoDisplay: true,
    editLink: false,
    copyright: "MIT Licensed | Copyright © 2023-Present 拂柳残声",
    displayFooter: true,
    fullscreen: true,
    pure: false,
    breadcrumb: true,
    blog: {
      name: "拂柳残声",
      avatar: "/portrait.jpg",
      roundAvatar: true,
      description: "倚天照海花无数，流水高山心自知。",
      intro: "https://fuliucansheng.github.io/",
      medias: {
        Github: "https://github.com/fuliucansheng",
        BiliBili: "https://space.bilibili.com/14477446",
        Email: "mailto:fuliucansheng@gmail.com",
        Weibo: "https://weibo.com/fuliucansheng",
        Zhihu: "https://www.zhihu.com/people/fuliucansheng",
        Twitter: "https://twitter.com/fuliucansheng",
        Youtube: "https://www.youtube.com/@fuliucansheng",
      },
    },
    blogLocales: {
      star: "精品",
    },
    plugins: {
      blog: {
        article: "/notes",
        category: "/notes/category",
        categoryItem: "/notes/category/:name/",
        tag: "/notes/tag/:name/",
        tagItem: "/notes/tag/:name/",
        star: "/notes-star",
        timeline: "/notes-timeline",
        filter: ({ filePathRelative, frontmatter }) => {
          if (!filePathRelative) return false;
          if (!filePathRelative.startsWith("notes/")) return false;
          if (frontmatter.home || frontmatter.layout) return false;

          return true;
        },
      },
      feed: {
        atom: false,
        json: false,
        rss: false,
      },
      mdEnhance: {
        tabs: true,
      codetabs: true,
      chart: true,
      echarts: true,
      mermaid: true,
      katex: true,
      mathjax: true,
      sub: true,
      sup: true,
      tasklist: true,
      card: true,
      figure: true,
      imgLazyload: true,
      imgMark: true,
      imgSize: true,
      attrs: true,
      presentation: true,
      mark: true,
      footnote: true,
      container: true,
      align: true,
      },
      photoSwipe: true,
      autoCatalog: true,
    },
  }),
  plugins: [
    shikiPlugin({
      theme: "dracula",
    }),
    searchProPlugin({
      indexContent: true,
      autoSuggestions: true,
      customFields: [
        {
          getter: (page) => page.frontmatter.category,
          formatter: "分类：$content",
        },
        {
          getter: (page) => page.frontmatter.tag,
          formatter: "标签：$content",
        },
      ],
    }),
  ],
});
