// Project rows that slide to their metrics: each row of project cards gets a toggle, and a second
// row of cards in the same shape slides in with that project's own numbers (injected server-side
// into window.KAYA_SHOWCASE — the browser never fetches anything itself) and how a request flows.
// A project with no counters of its own says so; it never borrows another project's numbers.
(function () {
  'use strict';
  var page = document.getElementById('page');
  if (!page) return;
  var still = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var data = window.KAYA_SHOWCASE || {};

  var LABELS = {
    en: { show: 'Metrics →', back: '← Projects', week: 'This week', none: 'No public counters yet',
          runs: 'Runs', flow: 'How a request flows', get: 'Get the prompt for this idea', copied: 'Copied' },
    pt: { show: 'Métricas →', back: '← Projetos', week: 'Esta semana', none: 'Ainda sem contadores públicos',
          runs: 'Corre', flow: 'Como corre um pedido', get: 'Obter o prompt desta ideia', copied: 'Copiado' }
  };

  var decks = [];
  var counter = 0;

  function element(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text) node.textContent = text;
    return node;
  }

  function formatValue(language, value) {
    try {
      return new Intl.NumberFormat(language === 'pt' ? 'pt-PT' : 'en', { notation: 'compact' }).format(value);
    } catch (e) {
      return String(Math.round(value));
    }
  }

  function ownStats(project) {
    var stats = ((data.products || {})[project] || {}).stats || [];
    var kept = [];
    stats.forEach(function (row) {
      if (typeof row.value === 'number') kept.push(row);
    });
    return kept;
  }

  function appendMeters(card, stats, language) {
    var max = 0;
    stats.forEach(function (row) { if (row.value > max) max = row.value; });
    var list = element('ul', 'meters');
    stats.forEach(function (row) {
      var line = element('li', 'meter');
      line.appendChild(element('span', 'meter-label', language === 'pt' ? row.pt : row.en));
      var track = element('span', 'meter-track');
      var fill = element('span', 'meter-fill');
      fill.style.setProperty('--v', String(max > 0 ? row.value / max : 0));
      track.appendChild(fill);
      line.appendChild(track);
      line.appendChild(element('b', '', formatValue(language, row.value)));
      list.appendChild(line);
    });
    card.appendChild(list);
  }

  function appendNoCounters(card, source, language) {
    var runs = source.querySelector('.more .runs');
    if (runs) {
      card.appendChild(element('p', 'more-label', LABELS[language].runs));
      card.appendChild(element('p', 'runs', runs.textContent));
    }
    card.appendChild(element('p', 'metric-none', LABELS[language].none));
  }

  function appendFlow(card, project, language) {
    var steps = ((data.flows || {})[project] || {})[language];
    if (!steps || !steps.length) return;
    card.appendChild(element('p', 'more-label', LABELS[language].flow));
    var list = element('ol', 'flow-steps');
    steps.forEach(function (stepText) { list.appendChild(element('li', 'flow-step', stepText)); });
    card.appendChild(list);
  }

  // A refused clipboard leaves the text on screen and selected, so it can still be copied by hand.
  function select(node) {
    try {
      var selection = window.getSelection();
      if (!selection || !node.firstChild) return;
      var range = document.createRange();
      range.setStartBefore(node.firstChild);
      range.setEndAfter(node.lastChild);
      selection.removeAllRanges();
      selection.addRange(range);
    } catch (e) {}
  }

  function appendPrompt(card, project, language) {
    if (project !== 'labwatch') return;
    var text = (data.prompt || {}).labwatch;
    if (!text) return;
    var button = element('button', 'prompt-btn', LABELS[language].get);
    button.setAttribute('type', 'button');
    card.appendChild(button);
    var box = element('pre', 'prompt-box', text);
    card.appendChild(box);
    button.addEventListener('click', function () {
      box.className = 'prompt-box show';
      try {
        navigator.clipboard.writeText(text).then(
            function () {
              button.textContent = LABELS[language].copied;
              window.setTimeout(function () { button.textContent = LABELS[language].get; }, 2000);
            },
            function () { select(box); });
      } catch (e) {
        select(box);
      }
    });
  }

  function metricCard(source, language) {
    var project = source.getAttribute('data-project') || '';
    var card = element('article', 'project metric-card');
    card.setAttribute('data-metrics', project);
    var title = source.querySelector('h3');
    card.appendChild(element('h3', '', title ? (title.getAttribute('data-text') || title.textContent.trim()) : project));
    var stats = ownStats(project);
    if (stats.length) {
      // The shared header only holds when every bar is week-scoped; a lifetime or derived bar carries
      // its window inside its own label instead, so a mixed card gets no "This week" header.
      var allWeek = stats.every(function (row) { return (row.scope || 'week') === 'week'; });
      if (allWeek) card.appendChild(element('p', 'more-label', LABELS[language].week));
      appendMeters(card, stats, language);
    } else {
      appendNoCounters(card, source, language);
    }
    appendFlow(card, project, language);
    appendPrompt(card, project, language);
    return card;
  }

  function hide(node, hidden) {
    if (hidden) {
      node.setAttribute('inert', '');
      node.setAttribute('aria-hidden', 'true');
    } else {
      node.removeAttribute('inert');
      node.removeAttribute('aria-hidden');
    }
  }

  function show(deck, metrics) {
    deck.metrics = metrics;
    deck.track.style.transform = metrics ? 'translate3d(-50%,0,0)' : 'translate3d(0,0,0)';
    hide(deck.projects, metrics);
    hide(deck.metricsGrid, !metrics);
    deck.toggle.setAttribute('aria-pressed', String(metrics));
    deck.toggle.textContent = metrics ? LABELS[deck.language].back : LABELS[deck.language].show;
  }

  function build(grid, language) {
    var cards = grid.querySelectorAll('.project');
    if (!cards.length) return;
    var deck = element('div', 'deck');
    var track = element('div', 'deck-track');
    grid.parentNode.insertBefore(deck, grid);
    deck.appendChild(track);
    track.appendChild(grid);
    var metricsGrid = element('div', 'projects projects-metrics');
    metricsGrid.id = 'metrics-' + language + '-' + (counter++);
    Array.prototype.forEach.call(cards, function (card) { metricsGrid.appendChild(metricCard(card, language)); });
    track.appendChild(metricsGrid);

    var toggle = element('button', 'deck-toggle');
    toggle.setAttribute('type', 'button');
    toggle.setAttribute('aria-controls', metricsGrid.id);
    var head = element('div', 'deck-head');
    var title = deck.previousElementSibling;
    if (title && title.classList.contains('group-title')) {
      title.parentNode.insertBefore(head, title);
      head.appendChild(title);
    } else {
      deck.parentNode.insertBefore(head, deck);
    }
    head.appendChild(toggle);

    var entry = { track: track, projects: grid, metricsGrid: metricsGrid, toggle: toggle,
                  language: language, metrics: false };
    toggle.addEventListener('click', function () { show(entry, !entry.metrics); });
    show(entry, false);
    decks.push(entry);
  }

  Array.prototype.forEach.call(page.querySelectorAll('.doc'), function (doc) {
    var language = doc.getAttribute('data-lang') === 'pt' ? 'pt' : 'en';
    Array.prototype.forEach.call(doc.querySelectorAll('.projects'), function (grid) { build(grid, language); });
  });

  document.addEventListener('kaya:lang', function () {
    decks.forEach(function (deck) { if (deck.metrics) show(deck, false); });
  });

  if (still) page.classList.add('decks-still');
})();
