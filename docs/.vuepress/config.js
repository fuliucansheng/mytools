import { defineUserConfig } from "@vuepress/cli";
import { shikiPlugin } from "@vuepress/plugin-shiki";
import { searchProPlugin } from "vuepress-plugin-search-pro";
import theme from "./theme.js";

export default defineUserConfig({
  lang: "zh-CN",
  title: "风中追寻の技术人生",
  description: "专注于互联网技术与人工智能",
  theme,
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
