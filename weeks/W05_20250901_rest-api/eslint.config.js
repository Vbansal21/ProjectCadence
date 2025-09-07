// eslint.config.js
import js from "@eslint/js";
import globals from "globals";
import jest from "eslint-plugin-jest";

export default [
  // What to ignore everywhere
  {
    ignores: [
      "node_modules/",
      "coverage/",
      "dist/",
      "**/*.min.js",
      // if you lint from monorepo root, ignore other weeks:
      // "../**/*"
    ],
  },

  // Base rules (Node ESM)
  {
    languageOptions: {
      ecmaVersion: "latest",
      sourceType: "module",
      globals: {
        ...globals.node,
      },
    },
    linterOptions: {
      reportUnusedDisableDirectives: true,
    },
    rules: {
      ...js.configs.recommended.rules,
      "no-unused-vars": ["warn", { argsIgnorePattern: "^_", varsIgnorePattern: "^_" }],
      "no-console": "off",
    },
  },

  // Browser console (public/) JS or inline scripts
  {
    files: ["public/**/*.js"],
    languageOptions: {
      globals: {
        ...globals.browser,
      },
    },
    rules: {},
  },

  // Tests (Jest)
  {
    files: ["tests/**/*.js"],
    plugins: { jest },
    languageOptions: {
      globals: {
        ...globals.node,
        ...globals.jest,
      },
    },
    rules: {
      ...jest.configs.recommended.rules,
    },
  },
];
