/* Dashboard charts (Chart.js). Data comes from the JSON block rendered by Flask. */
(function () {
  "use strict";

  var dataNode = document.getElementById("dashboard-data");
  var fallback = document.getElementById("chart-fallback");
  if (!dataNode) return;

  if (typeof Chart === "undefined") {
    if (fallback) fallback.classList.remove("d-none");
    return;
  }

  var data = JSON.parse(dataNode.textContent);
  var sym = data.symbol;
  var navy = "#12305c";
  var amber = "#e8a33d";

  function money(v) {
    return sym + Math.round(v).toLocaleString("en-US");
  }
  function compact(v) {
    return sym + (v / 1000).toLocaleString("en-US", { maximumFractionDigits: 0 }) + "k";
  }

  Chart.defaults.font.family = '"Segoe UI", system-ui, sans-serif';
  Chart.defaults.color = "#33445c";
  var common = { responsive: true, maintainAspectRatio: false };

  // 1. Price distribution
  new Chart(document.getElementById("chart-hist"), {
    type: "bar",
    data: {
      labels: data.histogram.labels,
      datasets: [{ label: "Houses", data: data.histogram.counts, backgroundColor: navy, borderRadius: 3 }]
    },
    options: Object.assign({}, common, {
      plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: true, text: "Price range" }, ticks: { maxRotation: 60, minRotation: 45, autoSkip: true } },
        y: { title: { display: true, text: "Number of houses" }, beginAtZero: true }
      }
    })
  });

  // 2. Property area vs price
  new Chart(document.getElementById("chart-scatter"), {
    type: "scatter",
    data: { datasets: [{ label: "House", data: data.scatter, backgroundColor: "rgba(18,48,92,0.5)", pointRadius: 3 }] },
    options: Object.assign({}, common, {
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: function (c) { return c.raw.x.toLocaleString("en-US") + " sq ft, " + money(c.raw.y); } } }
      },
      scales: {
        x: { title: { display: true, text: "Property area (sq ft)" } },
        y: { title: { display: true, text: "Price" }, ticks: { callback: compact } }
      }
    })
  });

  // 3. Model comparison (errors)
  new Chart(document.getElementById("chart-models"), {
    type: "bar",
    data: {
      labels: data.models.names,
      datasets: [
        { label: "MAE", data: data.models.mae, backgroundColor: navy, borderRadius: 3 },
        { label: "RMSE", data: data.models.rmse, backgroundColor: amber, borderRadius: 3 }
      ]
    },
    options: Object.assign({}, common, {
      plugins: { tooltip: { callbacks: { label: function (c) { return c.dataset.label + ": " + money(c.raw); } } } },
      scales: { y: { beginAtZero: true, title: { display: true, text: "Error" }, ticks: { callback: compact } } }
    })
  });

  // 4. Feature importance
  new Chart(document.getElementById("chart-importance"), {
    type: "bar",
    data: {
      labels: data.importance.map(function (i) { return i.label; }),
      datasets: [{ label: "Importance (%)", data: data.importance.map(function (i) { return i.value; }), backgroundColor: navy, borderRadius: 3 }]
    },
    options: Object.assign({}, common, {
      indexAxis: "y",
      plugins: { legend: { display: false } },
      scales: { x: { beginAtZero: true, title: { display: true, text: "Importance (%)" } } }
    })
  });
})();
