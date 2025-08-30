import * as d3 from 'd3';

const mount = document.getElementById('chart');
if (!mount) { console.warn('No #chart'); }
else {
  const data = [3,1,4,1,5,9];
  const w = 640, h = 280, m = {top:10,right:10,bottom:20,left:30};
  const x = d3.scaleBand().domain(data.map((_,i)=>String(i))).range([m.left, w-m.right]).padding(0.2);
  const y = d3.scaleLinear().domain([0, d3.max(data)!]).nice().range([h-m.bottom, m.top]);

  const svg = d3.select(mount).append('svg').attr('width', w).attr('height', h);
  svg.append('g').selectAll('rect').data(data).enter().append('rect')
    .attr('x',(_,i)=>x(String(i))!).attr('y',d=>y(d))
    .attr('width',x.bandwidth()).attr('height',d=>y(0)-y(d)).attr('rx',6)
    .attr('class','fill-neutral-300');

  svg.append('g').attr('transform',`translate(0,${h-m.bottom})`)
    .call(d3.axisBottom(x).tickSizeOuter(0)).attr('class','text-neutral-400');
  svg.append('g').attr('transform',`translate(${m.left},0)`)
    .call(d3.axisLeft(y).ticks(5)).attr('class','text-neutral-400');
}
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
