// The hero: Kaya and its agents as a turning constellation; a cursor blob reveals their names.
(function () {
  'use strict';
  var page = document.getElementById('page');
  var stage = document.querySelector('.stage');
  var canvas = stage ? stage.querySelector('canvas') : null;
  if (!page || !canvas) return;
  if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  var ctx = canvas.getContext('2d');
  if (!ctx) return;

  var TRAIL = 6;
  var nodes = [];
  var edges = [];
  var colors = {};
  var width = 0;
  var height = 0;
  var pointer = { x: 0, y: 0, seen: -1e9 };
  var tilt = { x: 0, y: 0 };
  var trail = [];
  var speed = 0;

  function readColors() {
    var style = getComputedStyle(document.documentElement);
    colors.ink = style.getPropertyValue('--ink').trim() || '#1b1f27';
    colors.teal = style.getPropertyValue('--teal').trim() || '#1d5c63';
    colors.amber = style.getPropertyValue('--amber').trim() || '#a85f16';
  }

  function labelOf(article) {
    var title = article.querySelector('h3');
    if (!title) return article.getAttribute('data-project');
    return title.getAttribute('data-text') || title.textContent.trim();
  }

  function sphere(count) {
    var points = [];
    var golden = Math.PI * (3 - Math.sqrt(5));
    for (var index = 0; index < count; index++) {
      var height01 = 1 - (index + 0.5) * 2 / count;
      var ring = Math.sqrt(1 - height01 * height01);
      var angle = golden * index;
      points.push({ x: Math.cos(angle) * ring, y: height01, z: Math.sin(angle) * ring });
    }
    return points;
  }

  function link() {
    edges = [];
    var seen = {};
    function add(from, to) {
      var key = Math.min(from, to) + '-' + Math.max(from, to);
      if (seen[key]) return;
      seen[key] = true;
      edges.push([from, to]);
    }
    for (var from = 1; from < nodes.length; from++) {
      add(0, from);
      var others = [];
      for (var to = 1; to < nodes.length; to++) {
        if (to === from) continue;
        var dx = nodes[from].x - nodes[to].x;
        var dy = nodes[from].y - nodes[to].y;
        var dz = nodes[from].z - nodes[to].z;
        others.push({ index: to, distance: dx * dx + dy * dy + dz * dz });
      }
      others.sort(function (left, right) { return left.distance - right.distance; });
      for (var rank = 0; rank < Math.min(2, others.length); rank++) add(from, others[rank].index);
    }
  }

  function build() {
    var lang = page.getAttribute('data-lang') || 'en';
    var articles = page.querySelectorAll('.doc[data-lang="' + lang + '"] article[data-project]');
    var points = sphere(articles.length);
    nodes = [{ x: 0, y: 0, z: 0, label: 'Kaya', core: true, phase: 0 }];
    for (var index = 0; index < articles.length; index++) {
      nodes.push({ x: points[index].x, y: points[index].y, z: points[index].z,
                   label: labelOf(articles[index]), core: false, phase: index * 1.7 });
    }
    link();
  }

  function resize() {
    var ratio = Math.min(window.devicePixelRatio || 1, 2);
    width = stage.clientWidth;
    height = stage.clientHeight;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    canvas.style.width = width + 'px';
    canvas.style.height = height + 'px';
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  }

  function project(node, time, spinY, spinX) {
    var drift = node.core ? 0 : 0.045;
    var x = node.x + Math.sin(time * 0.0007 + node.phase) * drift;
    var y = node.y + Math.cos(time * 0.0006 + node.phase) * drift;
    var z = node.z;
    var cosY = Math.cos(spinY), sinY = Math.sin(spinY);
    var turnedX = x * cosY + z * sinY;
    var turnedZ = -x * sinY + z * cosY;
    var cosX = Math.cos(spinX), sinX = Math.sin(spinX);
    var tippedY = y * cosX - turnedZ * sinX;
    var tippedZ = y * sinX + turnedZ * cosX;
    var radius = Math.min(width, height) * 0.32;
    var scale = 3.2 / (3.2 + tippedZ);
    return { x: width / 2 + turnedX * radius * scale, y: height * 0.5 + tippedY * radius * scale,
             z: tippedZ, scale: scale };
  }

  // The stage sits beside the name on wide screens (and above it on phones), so the
  // pointer is taken relative to the stage's own box, left and top.
  function aim(time, box) {
    if (time - pointer.seen < 2500) return { x: pointer.x - box.left, y: pointer.y - box.top };
    return { x: width * (0.5 + 0.3 * Math.cos(time * 0.00035)),
             y: height * (0.5 + 0.22 * Math.sin(time * 0.0005)) };
  }

  function follow(target) {
    if (!trail.length) {
      for (var fill = 0; fill < TRAIL; fill++) trail.push({ x: target.x, y: target.y });
    }
    var head = trail[0];
    var lastX = head.x, lastY = head.y;
    head.x += (target.x - head.x) * 0.22;
    head.y += (target.y - head.y) * 0.22;
    speed = Math.sqrt((head.x - lastX) * (head.x - lastX) + (head.y - lastY) * (head.y - lastY));
    for (var index = 1; index < trail.length; index++) {
      trail[index].x += (trail[index - 1].x - trail[index].x) * 0.35;
      trail[index].y += (trail[index - 1].y - trail[index].y) * 0.35;
    }
  }

  function blobPath() {
    var base = Math.min(width, height) * 0.11;
    var stretch = Math.min(1 + speed / 40, 1.6);
    ctx.beginPath();
    for (var index = 0; index < trail.length; index++) {
      var size = base * (1 - index / (TRAIL + 2)) * (index === 0 ? stretch : 1);
      ctx.moveTo(trail[index].x + size, trail[index].y);
      ctx.arc(trail[index].x, trail[index].y, size, 0, Math.PI * 2);
    }
  }

  function dot(point, color, size) {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(point.x, point.y, size * point.scale, 0, Math.PI * 2);
    ctx.fill();
  }

  function draw(time, spinY, spinX) {
    ctx.clearRect(0, 0, width, height);
    var placed = nodes.map(function (node) { return project(node, time, spinY, spinX); });
    var order = placed.map(function (point, index) { return index; })
      .sort(function (left, right) { return placed[right].z - placed[left].z; });
    ctx.globalAlpha = 0.22;
    ctx.strokeStyle = colors.ink;
    ctx.lineWidth = 1;
    ctx.beginPath();
    edges.forEach(function (edge) {
      ctx.moveTo(placed[edge[0]].x, placed[edge[0]].y);
      ctx.lineTo(placed[edge[1]].x, placed[edge[1]].y);
    });
    ctx.stroke();
    ctx.globalAlpha = 1;
    order.forEach(function (index) {
      dot(placed[index], nodes[index].core ? colors.amber : colors.teal, nodes[index].core ? 16 : 7);
    });
    blobPath();
    ctx.globalAlpha = 0.14;
    ctx.fillStyle = colors.teal;
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.save();
    ctx.clip();
    ctx.textBaseline = 'middle';
    order.forEach(function (index) {
      var node = nodes[index];
      var size = node.core ? 16 : 7;
      dot(placed[index], node.core ? colors.teal : colors.amber, size);
      ctx.fillStyle = colors.ink;
      ctx.font = (node.core ? '800 ' : '600 ') + (window.innerWidth < 640 ? 12 : 14) + 'px system-ui, sans-serif';
      // A name near the right edge goes on the node's left, so the stage never cuts it.
      var gap = size * placed[index].scale + 8;
      var left = placed[index].x + gap + ctx.measureText(node.label).width > width - 8;
      ctx.textAlign = left ? 'right' : 'left';
      ctx.fillText(node.label, placed[index].x + (left ? -gap : gap), placed[index].y);
    });
    ctx.restore();
  }

  function frame(time) {
    requestAnimationFrame(frame);
    var progress = Math.min(Math.max(window.scrollY / Math.max(height, 1), 0), 1);
    if (progress >= 1) return;
    var target = aim(time, stage.getBoundingClientRect());
    tilt.x += ((target.x / Math.max(width, 1) - 0.5) * 0.5 - tilt.x) * 0.05;
    tilt.y += ((target.y / Math.max(height, 1) - 0.5) * 0.35 - tilt.y) * 0.05;
    follow(target);
    draw(time, time * 0.00008 + progress * Math.PI * 1.2 + tilt.x, 0.3 + progress * 0.6 + tilt.y);
  }

  window.addEventListener('pointermove', function (event) {
    pointer.x = event.clientX;
    pointer.y = event.clientY;
    pointer.seen = performance.now();
  }, { passive: true });
  window.addEventListener('resize', resize);
  document.addEventListener('kaya:lang', build);
  document.addEventListener('kaya:scheme', readColors);
  if (window.matchMedia) {
    var dark = window.matchMedia('(prefers-color-scheme: dark)');
    if (dark.addEventListener) dark.addEventListener('change', readColors);
  }
  page.classList.add('scene-on');
  readColors();
  build();
  resize();
  requestAnimationFrame(frame);
})();
