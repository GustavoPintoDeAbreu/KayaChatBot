// Page motion: console greeting, letter-by-letter hover text, smooth wheel scrolling, scroll-scrubbed reveals.
(function () {
  'use strict';
  var page = document.getElementById('page');
  if (!page) return;
  var still = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function greet() {
    var style = getComputedStyle(document.documentElement);
    var teal = style.getPropertyValue('--teal').trim() || '#1d5c63';
    var amber = style.getPropertyValue('--amber').trim() || '#a85f16';
    console.log('%cKaya', 'font: 800 44px system-ui, sans-serif; color: ' + teal + ';');
    console.log('%cA home lab of AI agents. They planned and built this page; a person said yes.',
                'font: 600 13px system-ui, sans-serif; color: ' + amber + ';');
  }

  function split(element) {
    if (element.getAttribute('data-text') !== null) return;
    var text = element.textContent.trim();
    if (!text) return;
    element.setAttribute('data-text', text);
    while (element.firstChild) element.removeChild(element.firstChild);
    var reader = document.createElement('span');
    reader.className = 'sr-only';
    reader.textContent = text;
    element.appendChild(reader);
    var count = 0;
    text.split(/\s+/).forEach(function (word, position) {
      if (!word) return;
      if (position > 0) element.appendChild(document.createTextNode(' '));
      var holder = document.createElement('span');
      holder.className = 'split-word';
      holder.setAttribute('aria-hidden', 'true');
      Array.from(word).forEach(function (letter) {
        var cell = document.createElement('span');
        cell.className = 'split-ch';
        cell.style.setProperty('--i', String(count++));
        var first = document.createElement('span');
        first.textContent = letter;
        var second = document.createElement('span');
        second.textContent = letter;
        cell.appendChild(first);
        cell.appendChild(second);
        holder.appendChild(cell);
      });
      element.appendChild(holder);
    });
  }

  function smoothWheel() {
    var target = 0;
    var current = 0;
    var active = false;
    function limit() { return Math.max(0, document.documentElement.scrollHeight - window.innerHeight); }
    function step() {
      if (!active) return;
      current += (target - current) * 0.12;
      if (Math.abs(target - current) < 0.5) { current = target; active = false; }
      window.scrollTo(0, current);
      if (active) requestAnimationFrame(step);
    }
    window.addEventListener('wheel', function (event) {
      if (event.ctrlKey || event.defaultPrevented || Math.abs(event.deltaX) > Math.abs(event.deltaY)) return;
      var unit = event.deltaMode === 1 ? 16 : (event.deltaMode === 2 ? window.innerHeight : 1);
      if (!active) { current = window.scrollY; target = current; }
      target = Math.min(Math.max(target + event.deltaY * unit, 0), limit());
      event.preventDefault();
      if (!active) { active = true; requestAnimationFrame(step); }
    }, { passive: false });
    ['keydown', 'pointerdown', 'touchstart'].forEach(function (name) {
      window.addEventListener(name, function () { active = false; }, { passive: true });
    });
  }

  function scrub() {
    var items = Array.prototype.map.call(page.querySelectorAll('[data-scrub]'), function (element) {
      return { element: element, kind: element.getAttribute('data-scrub'), shift: 0 };
    });
    var queued = false;
    function update() {
      queued = false;
      var view = window.innerHeight;
      var tops = items.map(function (item) { return item.element.getBoundingClientRect().top - item.shift; });
      items.forEach(function (item, index) {
        var rest = 1 - Math.min(Math.max((view - tops[index]) / (view * 0.5), 0), 1);
        if (item.kind === 'slide') {
          item.element.style.transform = 'translate3d(' + (-rest * window.innerWidth * 0.08).toFixed(1) + 'px,0,0)';
        } else {
          item.shift = Math.round(rest * 560) / 10;
          item.element.style.transform = 'translate3d(0,' + item.shift + 'px,0)';
        }
      });
    }
    function queue() {
      if (queued) return;
      queued = true;
      requestAnimationFrame(update);
    }
    window.addEventListener('scroll', queue, { passive: true });
    window.addEventListener('resize', queue);
    document.addEventListener('kaya:lang', queue);
    update();
  }

  greet();
  if (still) return;
  Array.prototype.forEach.call(page.querySelectorAll('[data-split]'), split);
  smoothWheel();
  scrub();
})();
