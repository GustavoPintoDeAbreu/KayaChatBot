// Full detail views: clicking a project card slides open an overlay with the card's own details,
// the lab's live numbers (injected server-side into window.KAYA_SHOWCASE — the browser never
// fetches anything itself) and how a request flows. Keyboard-accessible; nothing here is required to read.
(function () {
  'use strict';
  var page = document.getElementById('page');
  if (!page) return;
  var still = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var data = window.KAYA_SHOWCASE || {};

  var LABELS = {
    en: { close: 'Close', week: 'This week in the lab', fit: 'How it fits together',
          get: 'Get the prompt for this idea', copied: 'Copied' },
    pt: { close: 'Fechar', week: 'Esta semana no lab', fit: 'Como encaixa tudo',
          get: 'Obter o prompt desta ideia', copied: 'Copiado' }
  };

  function lang() { return page.getAttribute('data-lang') === 'pt' ? 'pt' : 'en'; }

  function formatValue(language, unit, value) {
    if (typeof value !== 'number') return '';
    var locale = language === 'pt' ? 'pt-PT' : 'en';
    try {
      if (unit === '€') {
        return '€' + new Intl.NumberFormat(locale, { minimumFractionDigits: 2, maximumFractionDigits: 2 })
          .format(value);
      }
      if (unit === ' kWh') {
        return new Intl.NumberFormat(locale, { minimumFractionDigits: 1, maximumFractionDigits: 1 })
          .format(value) + ' kWh';
      }
      return new Intl.NumberFormat(locale, { notation: 'compact' }).format(value);
    } catch (e) {
      return String(Math.round(value));
    }
  }

  var panel = document.createElement('div');
  panel.className = 'detail-panel';
  panel.setAttribute('role', 'dialog');
  panel.setAttribute('aria-modal', 'true');
  var closeBtn = document.createElement('button');
  closeBtn.setAttribute('type', 'button');
  closeBtn.className = 'detail-close';
  var body = document.createElement('div');
  body.className = 'detail-body';
  panel.appendChild(closeBtn);
  panel.appendChild(body);
  page.appendChild(panel);

  var open = false;
  var origin = null;

  function heading(text) {
    var line = document.createElement('p');
    line.className = 'detail-label';
    line.textContent = text;
    body.appendChild(line);
  }

  function appendStats(project, language) {
    var stats = ((data.products || {})[project] || {}).stats;
    if (!stats) stats = (data.lab || {}).stats;
    if (!stats || !stats.length) return;
    heading(LABELS[language].week);
    var list = document.createElement('ul');
    list.className = 'meters';
    var max = 0;
    stats.forEach(function (row) {
      if (typeof row.value === 'number' && row.value > max) max = row.value;
    });
    stats.forEach(function (row) {
      var line = document.createElement('li');
      line.className = 'meter';
      var label = document.createElement('span');
      label.textContent = language === 'pt' ? row.pt : row.en;
      line.appendChild(label);
      var track = document.createElement('span');
      track.className = 'meter-track';
      var fill = document.createElement('span');
      fill.className = 'meter-fill';
      fill.style.setProperty('--v', String(max > 0 && typeof row.value === 'number' ? row.value / max : 0));
      track.appendChild(fill);
      line.appendChild(track);
      var value = document.createElement('b');
      value.textContent = formatValue(language, row.unit, row.value);
      line.appendChild(value);
      list.appendChild(line);
    });
    body.appendChild(list);
  }

  function appendFlow(project, language) {
    var steps = ((data.flows || {})[project] || {})[language];
    if (!steps || !steps.length) return;
    heading(LABELS[language].fit);
    var list = document.createElement('ol');
    list.className = 'flow-steps';
    steps.forEach(function (stepText) {
      var step = document.createElement('li');
      step.className = 'flow-step';
      step.textContent = stepText;
      list.appendChild(step);
    });
    body.appendChild(list);
  }

  function appendPrompt(project, language) {
    if (project !== 'labwatch') return;
    var text = (data.prompt || {}).labwatch;
    if (!text) return;
    var button = document.createElement('button');
    button.setAttribute('type', 'button');
    button.className = 'prompt-btn';
    button.textContent = LABELS[language].get;
    body.appendChild(button);
    var box = document.createElement('pre');
    box.className = 'prompt-box';
    box.textContent = text;
    body.appendChild(box);
    button.addEventListener('click', function () {
      box.className = 'prompt-box show';
      try {
        navigator.clipboard.writeText(text);
        button.textContent = LABELS[language].copied;
        window.setTimeout(function () { button.textContent = LABELS[language].get; }, 2000);
      } catch (e) {}
    });
  }

  function renderCard(card) {
    var language = lang();
    clearBody();
    var more = card.querySelector('.more');
    if (more) body.appendChild(more.cloneNode(true));
    var project = card.getAttribute('data-project') || '';
    appendStats(project, language);
    appendFlow(project, language);
    appendPrompt(project, language);
    closeBtn.textContent = '×';
    closeBtn.setAttribute('aria-label', LABELS[language].close);
  }

  function clearBody() {
    while (body.firstChild) body.removeChild(body.firstChild);
  }

  function openPanel(card) {
    open = true;
    origin = card;
    renderCard(card);
    panel.classList.add('open');
    if (still) panel.style.transform = 'translate3d(0,0,0)';
    try { closeBtn.focus(); } catch (e) {}
  }

  function closePanel() {
    open = false;
    panel.classList.remove('open');
    if (still) panel.style.transform = 'translate3d(100%,0,0)';
    var back = origin;
    origin = null;
    if (back) { try { back.focus(); } catch (e) {} }
  }

  document.addEventListener('click', function (event) {
    if (!event.target.closest) return;
    if (open) {
      if (event.target.closest('.detail-close') ||
          (event.target.closest('.detail-panel') && !event.target.closest('.detail-body'))) closePanel();
      return;
    }
    var card = event.target.closest('.project');
    if (card) openPanel(card);
  });

  document.addEventListener('keydown', function (event) {
    if (open) {
      if (event.key === 'Escape') closePanel();
      return;
    }
    if (event.key !== 'Enter' && event.key !== ' ') return;
    if (!event.target.closest) return;
    var card = event.target.closest('.project');
    if (!card) return;
    if (event.key === ' ') event.preventDefault();
    openPanel(card);
  });

  document.addEventListener('kaya:lang', function () {
    if (open && origin) renderCard(origin);
  });
})();
