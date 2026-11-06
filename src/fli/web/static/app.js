// Frontier Lab Intelligence - progressive enhancement only; every page works without JS.
(function () {
  "use strict";

  // light / dark toggle (the head script applies the saved choice before first paint)
  var toggle = document.querySelector("[data-theme-toggle]");
  if (toggle) {
    toggle.addEventListener("click", function () {
      var root = document.documentElement;
      var dark = root.dataset.theme ? root.dataset.theme === "dark"
        : window.matchMedia("(prefers-color-scheme: dark)").matches;
      root.dataset.theme = dark ? "light" : "dark";
      try { localStorage.setItem("fli-theme", root.dataset.theme); } catch (e) { /* private mode */ }
    });
  }

  // filter forms submit as soon as a select or checkbox changes
  document.querySelectorAll("form[data-autosubmit]").forEach(function (form) {
    form.querySelectorAll("select, input[type=checkbox]").forEach(function (el) {
      el.addEventListener("change", function () { form.submit(); });
    });
  });

  // labelling queue: 0 / 1 / 2 label, s skip, r reason, Esc leaves a text field
  var labelForm = document.querySelector("form[data-label-form]");
  if (labelForm) {
    document.addEventListener("keydown", function (e) {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      var typing = /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName);
      if (typing) {
        if (e.key === "Escape") e.target.blur();
        return;
      }
      var button = labelForm.querySelector('button[data-key="' + e.key + '"]');
      if (button) { e.preventDefault(); button.click(); return; }
      if (e.key === "s") {
        var skip = labelForm.querySelector("[data-skip]");
        if (skip) { e.preventDefault(); window.location.href = skip.href; }
      } else if (e.key === "r") {
        var reason = labelForm.querySelector("[data-reason]");
        if (reason) { e.preventDefault(); reason.focus(); }
      }
    });
  }
})();
