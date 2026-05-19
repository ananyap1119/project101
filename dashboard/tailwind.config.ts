import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#111827",
        panel: "#f8fafc",
        line: "#d8dee9",
        mint: "#0f766e",
        coral: "#b45309",
      },
    },
  },
  plugins: [],
} satisfies Config;
