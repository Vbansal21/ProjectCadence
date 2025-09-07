import js from "@eslint/js";
import globals from "globals";
let jestPlugin;
try {
  const m = await import("eslint-plugin-jest");
  jestPlugin = m.default ?? m;
} catch {
  jestPlugin = null;
}

export default [
  {
    ignores: [
      "node_modules/",
      "coverage/",
      "dist/",
      "**/*.min.js",
    ],
  },

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

  // Tests: enable jest globals; add plugin rules only if available
  {
    files: ["tests/**/*.js"],
    plugins: jestPlugin ? { jest: jestPlugin } : {},
    languageOptions: { globals: { ...globals.node, ...globals.jest } },
    rules: jestPlugin ? { ...jestPlugin.configs.recommended.rules } : {}
  },
];
