"use strict";
(() => {
  // src/ts/site.ts
  function prefersDark() {
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  }
  document.addEventListener("htmx:responseError", () => {
    const el = document.createElement("div");
    el.className = "fixed bottom-4 right-4 rounded-lg bg-red-600/90 px-3 py-2 text-sm";
    el.textContent = "Load failed. Try again.";
    document.body.appendChild(el);
    setTimeout(() => el.remove(), 2500);
  });
})();
// Minimal HTMX logging
document.addEventListener('htmx:beforeRequest', (e) => {
  console.log('[HTMX] GET ->', e.detail.path, e.detail.verb);
});
document.addEventListener('htmx:responseError', (e) => {
  console.error('[HTMX] ERROR', e.detail.xhr.status, e.detail.xhr.responseURL);
});
