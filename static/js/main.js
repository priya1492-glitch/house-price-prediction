/* Site-wide behaviour: prediction form validation and the demo contact form. */
(function () {
  "use strict";

  /* ---------- Prediction form ---------- */
  var predictForm = document.getElementById("predict-form");
  if (predictForm) {
    var submitButton = document.getElementById("predict-submit");

    function checkField(field) {
      var ok = field.checkValidity();
      var value = field.value.trim();

      if (ok && field.type === "number" && value !== "" && field.step === "1" && !/^-?\d+$/.test(value)) {
        field.setCustomValidity("Enter a whole number.");
        ok = false;
      } else {
        field.setCustomValidity("");
      }
      field.classList.toggle("is-invalid", !ok);
      return ok;
    }

    predictForm.addEventListener("submit", function (event) {
      var fields = predictForm.querySelectorAll("input[name]:not([type=hidden]), select[name]");
      var firstInvalid = null;

      fields.forEach(function (field) {
        if (!checkField(field) && !firstInvalid) firstInvalid = field;
      });

      if (firstInvalid) {
        event.preventDefault();
        firstInvalid.focus();
        return;
      }

      // Loading state while the server calculates the estimate.
      if (submitButton) {
        submitButton.classList.add("is-loading");
        submitButton.setAttribute("aria-busy", "true");
        var label = submitButton.querySelector(".btn-text");
        if (label) label.textContent = "Estimating...";
        // Let the browser submit first, then disable to avoid double posts.
        setTimeout(function () { submitButton.disabled = true; }, 0);
      }
    });

    predictForm.addEventListener("input", function (event) {
      if (event.target.classList.contains("is-invalid")) checkField(event.target);
    });

    // Bring the result or error into view after a submission.
    var target = document.querySelector("#form-error, #result-panel .result-price");
    if (target && window.location.hash === "" && window.matchMedia("(max-width: 991px)").matches) {
      target.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }

  /* ---------- Hero quick estimate ---------- */
  var quickForm = document.getElementById("quick-estimate-form");
  if (quickForm) {
    var quickSubmit = document.getElementById("quick-estimate-submit");
    var quickFields = quickForm.querySelectorAll("input[name]:not([type=hidden]), select[name]");
    var quickBhk = document.getElementById("quick-bhk-input");
    var quickBath = document.getElementById("quick-bath");

    function checkQuickField(field) {
      var ok = field.checkValidity();
      var value = field.value.trim();

      if (ok && field.type === "number" && value !== "" && field.step === "1" && !/^-?\d+$/.test(value)) {
        field.setCustomValidity("Enter a whole number.");
        ok = false;
      } else {
        field.setCustomValidity("");
      }
      field.classList.toggle("is-invalid", !ok);
      return ok;
    }

    function syncQuickBath() {
      if (!quickBhk || !quickBath) return;
      var bhk = Number(quickBhk.value);
      var maxBath = Number(quickBath.getAttribute("data-bath-max"));
      if (Number.isFinite(bhk) && Number.isFinite(maxBath)) {
        quickBath.value = String(Math.min(Math.max(1, bhk), maxBath));
      }
    }

    if (quickBhk) quickBhk.addEventListener("input", syncQuickBath);
    syncQuickBath();

    quickForm.addEventListener("submit", function (event) {
      var firstInvalid = null;
      quickFields.forEach(function (field) {
        if (!checkQuickField(field) && !firstInvalid) firstInvalid = field;
      });

      if (firstInvalid) {
        event.preventDefault();
        firstInvalid.focus();
        return;
      }

      if (quickSubmit) {
        quickSubmit.classList.add("is-loading");
        quickSubmit.setAttribute("aria-busy", "true");
        var label = quickSubmit.querySelector(".btn-text");
        if (label) label.textContent = "Estimating...";
        setTimeout(function () { quickSubmit.disabled = true; }, 0);
      }
    });

    quickForm.addEventListener("input", function (event) {
      if (event.target.classList.contains("is-invalid")) checkQuickField(event.target);
    });
    quickForm.addEventListener("change", function (event) {
      if (event.target.classList.contains("is-invalid")) checkQuickField(event.target);
    });
  }

  /* ---------- Scroll reveal ---------- */
  var revealItems = document.querySelectorAll("[data-reveal]");
  var reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (revealItems.length && !reducedMotion && "IntersectionObserver" in window) {
    var revealObserver = new IntersectionObserver(function (entries, observer) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.12 });

    revealItems.forEach(function (item) {
      item.classList.add("reveal-ready");
      revealObserver.observe(item);
    });
  }

  /* ---------- Contact form (demonstration only) ---------- */
  var contactForm = document.getElementById("contact-form");
  if (contactForm) {
    var feedback = document.getElementById("contact-feedback");
    var emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/;

    contactForm.addEventListener("submit", function (event) {
      event.preventDefault();
      var firstInvalid = null;

      contactForm.querySelectorAll("input, textarea").forEach(function (field) {
        var ok = field.checkValidity() && field.value.trim() !== "";
        if (ok && field.type === "email") ok = emailPattern.test(field.value.trim());
        field.classList.toggle("is-invalid", !ok);
        if (!ok && !firstInvalid) firstInvalid = field;
      });

      feedback.className = "mt-3";
      if (firstInvalid) {
        feedback.classList.add("text-danger");
        feedback.textContent = "Please correct the highlighted fields.";
        firstInvalid.focus();
        return;
      }

      feedback.classList.add("text-success");
      feedback.textContent = "Thanks! Your details look valid. This is a demonstration form, so nothing was sent.";
      contactForm.reset();
    });

    contactForm.addEventListener("input", function (event) {
      event.target.classList.remove("is-invalid");
    });
  }
})();
