/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: {
        display: ["Sora", "ui-sans-serif", "system-ui"],
      },
      boxShadow: {
        neon: "0 0 24px rgba(56, 189, 248, 0.45)",
      },
      animation: {
        pulseGlow: "pulseGlow 2.2s ease-in-out infinite",
      },
      keyframes: {
        pulseGlow: {
          "0%, 100%": { boxShadow: "0 0 0 rgba(0,0,0,0)" },
          "50%": { boxShadow: "0 0 24px rgba(16,185,129,0.5)" },
        },
      },
    },
  },
  plugins: [],
}
