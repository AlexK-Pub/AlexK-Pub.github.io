/* Theme toggle, e-mail link, BibTeX copy buttons, TIDE/CTM figure swap. No dependencies. */
(function () {
  "use strict";
  var root = document.documentElement;

  /* ---- theme ---- */
  function currentTheme() { return root.getAttribute("data-theme") === "light" ? "light" : "dark"; }
  function labelToggle(btn) {
    var next = currentTheme() === "dark" ? "light" : "dark";
    btn.setAttribute("aria-label", "Switch to " + next + " theme");
    btn.setAttribute("title", "Switch to " + next + " theme");
  }
  var toggle = document.querySelector("[data-theme-toggle]");
  if (toggle) {
    labelToggle(toggle);
    toggle.addEventListener("click", function () {
      var next = currentTheme() === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("theme", next); } catch (e) { /* storage unavailable: the choice lasts for this page */ }
      var meta = document.querySelector('meta[name="theme-color"]');
      if (meta) { meta.setAttribute("content", next === "light" ? "#F6F8FA" : "#0C1016"); }
      labelToggle(toggle);
    });
  }
  /* ---- e-mail: assembled here so the address is not in the page source as one string ---- */
  Array.prototype.forEach.call(document.querySelectorAll("[data-mail-user]"), function (el) {
    var addr = el.getAttribute("data-mail-user") + "@" + el.getAttribute("data-mail-domain");
    var a = document.createElement("a");
    a.href = "mailto:" + addr;
    a.textContent = addr;
    el.textContent = "";
    el.appendChild(a);
  });

  /* ---- copy buttons ---- */
  Array.prototype.forEach.call(document.querySelectorAll("[data-copy]"), function (btn) {
    var label = btn.textContent;
    btn.addEventListener("click", function () {
      var target = document.getElementById(btn.getAttribute("data-copy"));
      if (!target) { return; }
      var text = target.textContent;
      var done = function () { btn.textContent = "Copied"; setTimeout(function () { btn.textContent = label; }, 1800); };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, function () { selectNode(target); });
      } else { selectNode(target); }
    });
  });
  function selectNode(node) {
    var range = document.createRange(); range.selectNodeContents(node);
    var sel = window.getSelection(); sel.removeAllRanges(); sel.addRange(range);
  }

  /* ---- swap: two figures drawn on the same grid, shown one at a time ---- */
  Array.prototype.forEach.call(document.querySelectorAll("[data-swap]"), function (swap) {
    var buttons = swap.querySelectorAll("[data-show]");
    var panels = swap.querySelectorAll("[data-panel]");
    if (!buttons.length || !panels.length) { return; }
    function show(name) {
      Array.prototype.forEach.call(panels, function (p) { p.hidden = p.getAttribute("data-panel") !== name; });
      Array.prototype.forEach.call(buttons, function (b) { b.setAttribute("aria-pressed", String(b.getAttribute("data-show") === name)); });
    }
    Array.prototype.forEach.call(buttons, function (b) {
      b.addEventListener("click", function () { show(b.getAttribute("data-show")); });
    });
    swap.classList.add("is-enhanced");
    show(buttons[0].getAttribute("data-show"));
  });
})();
