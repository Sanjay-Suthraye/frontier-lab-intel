// Frontier Lab Intelligence - progressive enhancement only; every page works without JS.
(function () {
  "use strict";

  // filter forms submit as soon as a select, checkbox or window button changes; choosing a rolling
  // window (7 days / 30 days / whole window) clears the calendar-week choice, which would win
  document.querySelectorAll("form[data-autosubmit]").forEach(function (form) {
    form.querySelectorAll("select, input[type=checkbox], input[type=radio]").forEach(function (el) {
      el.addEventListener("change", function () {
        if (el.type === "radio" && el.name === "week") {
          var cw = form.querySelector("select[name=cw]");
          if (cw) cw.value = "";
        }
        form.submit();
      });
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
