let gaugeChart, barChart, shapChart;

function validate() {
  const age          = parseFloat(document.getElementById("age").value);
  const income       = parseFloat(document.getElementById("income").value);
  const totalYears   = parseFloat(document.getElementById("totalYears").value);
  const roleYears    = parseFloat(document.getElementById("roleYears").value);
  const managerYears = parseFloat(document.getElementById("managerYears").value);
  const errors = [];

  if (isNaN(age)    || age < 18 || age > 65)             errors.push("Age must be between 18 and 65.");
  if (isNaN(income) || income <= 0)                       errors.push("Monthly Income must be a positive number.");
  if (isNaN(totalYears) || totalYears < 0 || totalYears > 40) errors.push("Total Working Years must be between 0 and 40.");
  if (isNaN(roleYears)    || roleYears < 0)               errors.push("Years In Current Role cannot be negative.");
  if (isNaN(managerYears) || managerYears < 0)            errors.push("Years With Current Manager cannot be negative.");
  if (!isNaN(age) && !isNaN(totalYears)) {
    if (totalYears >= age)           errors.push("Total Working Years must be less than Age.");
    if ((age - totalYears) < 18)     errors.push("Total Working Years is unrealistic for the given Age.");
  }
  if (!isNaN(totalYears) && !isNaN(roleYears) && roleYears > totalYears)
    errors.push("Years In Current Role cannot exceed Total Working Years.");
  if (!isNaN(roleYears) && !isNaN(managerYears) && managerYears > roleYears)
    errors.push("Years With Current Manager cannot exceed Years In Current Role.");
  return errors;
}

document.getElementById("predictForm").addEventListener("submit", function(e) {
  e.preventDefault();
  const errorBox = document.getElementById("errorBox");
  const errors   = validate();
  if (errors.length) {
    errorBox.innerHTML = errors.map(m => `<p>${m}</p>`).join("");
    errorBox.classList.add("visible");
    return;
  }
  errorBox.classList.remove("visible");
  errorBox.innerHTML = "";

  const btn = document.getElementById("submitBtn");
  btn.disabled = true;
  btn.textContent = "Analyzing…";

  fetch("/predict", { method: "POST", body: new FormData(this) })
    .then(r => r.json())
    .then(data => {
      btn.disabled = false;
      btn.textContent = "Run Prediction";
      if (!data.success) {
        const msgs = data.errors ? data.errors.map(m => `<p>${m}</p>`).join("") : "<p>Prediction failed. Please try again.</p>";
        errorBox.innerHTML = msgs;
        errorBox.classList.add("visible");
        return;
      }
      renderResults(data);
    })
    .catch(() => {
      btn.disabled = false;
      btn.textContent = "Run Prediction";
      errorBox.innerHTML = "<p>Network error. Please check your connection.</p>";
      errorBox.classList.add("visible");
    });
});

function renderResults(data) {
  document.getElementById("emptyState").style.display = "none";
  document.getElementById("results").classList.add("visible");
  const isLeave = data.prediction === 1;
  const ap = data.attrition_prob;
  const rp = data.retention_prob;

  const banner = document.getElementById("verdictBanner");
  banner.className = "verdict " + (isLeave ? "leave" : "stay");
  document.getElementById("verdictIcon").textContent  = isLeave ? "⚠️" : "✅";
  document.getElementById("verdictLabel").textContent = isLeave ? "High Risk" : "Low Risk";
  document.getElementById("verdictTitle").textContent = isLeave ? "Employee Likely to Leave" : "Employee Likely to Stay";
  document.getElementById("verdictSub").textContent   = isLeave ? "Recommend immediate retention review." : "Retention profile looks healthy.";
  document.getElementById("verdictProb").textContent  = ap + "%";

  document.getElementById("attritionVal").textContent = ap + "%";
  document.getElementById("retentionVal").textContent = rp + "%";
  setTimeout(() => {
    document.getElementById("attritionBar").style.width = ap + "%";
    document.getElementById("retentionBar").style.width = rp + "%";
  }, 60);

  drawGauge(ap);
  drawBar(ap, rp);
  drawShap(data.top_features);
}

function drawGauge(value) {
  if (gaugeChart) gaugeChart.destroy();
  gaugeChart = new Chart(document.getElementById("gaugeChart"), {
    type: "doughnut",
    data: {
      labels: ["Attrition Risk", "Retention"],
      datasets: [{ data: [value, 100 - value], backgroundColor: ["#f43f5e","#10b981"], borderWidth: 0, hoverOffset: 4 }]
    },
    options: {
      cutout: "72%", responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { position: "bottom", labels: { color: "#6b7280", font: { family: "DM Sans", size: 11 }, padding: 14, boxWidth: 10 } },
        tooltip: { callbacks: { label: ctx => ` ${ctx.label}: ${ctx.parsed}%` } }
      }
    }
  });
}

function drawBar(ap, rp) {
  if (barChart) barChart.destroy();
  barChart = new Chart(document.getElementById("barChart"), {
    type: "bar",
    data: {
      labels: ["Attrition", "Retention"],
      datasets: [{ data: [ap, rp], backgroundColor: ["rgba(244,63,94,0.75)","rgba(16,185,129,0.75)"], borderRadius: 6, borderSkipped: false }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#6b7280", font: { family: "DM Sans", size: 11 } }, grid: { display: false }, border: { display: false } },
        y: { ticks: { color: "#6b7280", font: { family: "DM Mono", size: 10 }, callback: v => v + "%" }, grid: { color: "rgba(255,255,255,0.04)" }, border: { display: false }, min: 0, max: 100 }
      }
    }
  });
}

function drawShap(features) {
  if (shapChart) shapChart.destroy();
  const labels = features.map(f => f.feature);
  const values = features.map(f => f.value);
  const colors = values.map(v => v >= 0 ? "rgba(244,63,94,0.8)" : "rgba(16,185,129,0.8)");
  shapChart = new Chart(document.getElementById("shapChart"), {
    type: "bar",
    data: { labels, datasets: [{ label: "SHAP Impact", data: values, backgroundColor: colors, borderRadius: 4, borderSkipped: false }] },
    options: {
      indexAxis: "y", responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#6b7280", font: { family: "DM Mono", size: 10 } }, grid: { color: "rgba(255,255,255,0.04)" }, border: { display: false }, title: { display: true, text: "Impact on attrition probability", color: "#6b7280", font: { size: 11 } } },
        y: { ticks: { color: "#e8eaf0", font: { family: "DM Sans", size: 12 } }, grid: { display: false }, border: { display: false } }
      }
    }
  });
}
