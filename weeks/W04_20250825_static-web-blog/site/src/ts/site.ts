// Dark-mode toggle stub (extend later)
export function prefersDark(): boolean {
  return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
}

// Optional HTMX: show a friendly message on error
document.addEventListener('htmx:responseError', () => {
  const el = document.createElement('div');
  el.className = 'fixed bottom-4 right-4 rounded-lg bg-red-600/90 px-3 py-2 text-sm';
  el.textContent = 'Load failed. Try again.';
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 2500);
});
