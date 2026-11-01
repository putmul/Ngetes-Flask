const predictionForm = document.getElementById("prediction-form");
const predictionResult = document.getElementById("prediction-result");
const gramResult = document.getElementById("gram-result");
const predictionPeriod = document.getElementById("prediction-period");
const predictionError = document.getElementById("prediction-error");
const selectedPeriod = document.getElementById("selected-period");
const monthSlider = document.getElementById("month-slider");
const selectedMonthName = document.getElementById("selected-month-name");
const selectedMonthNumber = document.getElementById("selected-month-number");

const months = [
  "Januari", "Februari", "Maret", "April", "Mei", "Juni",
  "Juli", "Agustus", "September", "Oktober", "November", "Desember",
];
const years = [2026, 2027, 2028, 2029, 2030];
let selectedYear = 2026;
let selectedMonth = 8;

function updateSelection() {
  document.querySelectorAll("#year-options button").forEach((button) => {
    button.classList.toggle("active", Number(button.dataset.year) === selectedYear);
  });
  monthSlider.value = selectedMonth;
  selectedMonthName.textContent = months[selectedMonth - 1];
  selectedMonthNumber.textContent = `Bulan ke-${selectedMonth}`;
  const isProjection = selectedYear > 2026 || (selectedYear === 2026 && selectedMonth > 8);
  selectedPeriod.textContent = `${months[selectedMonth - 1]} ${selectedYear} · ${isProjection ? "Periode proyeksi" : "Periode data training"}`;
}

function renderPeriodSelectors() {
  const yearOptions = document.getElementById("year-options");
  years.forEach((year) => {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.year = year;
    button.textContent = year;
    button.addEventListener("click", () => { selectedYear = year; updateSelection(); });
    yearOptions.appendChild(button);
  });
  updateSelection();
}

async function convertToGram(predictionValue) {
  const response = await fetch(`/api/gram-convert?prediction=${encodeURIComponent(predictionValue)}`);
  const info = await response.json();
  if (!response.ok) throw new Error(info.error || "Konversi ke gram gagal.");
  return info;
}

predictionForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  predictionError.hidden = true;
  predictionResult.textContent = "Menghitung...";
  gramResult.textContent = "Menghitung...";
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ year: selectedYear, month: selectedMonth }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Prediksi gagal.");
    predictionResult.textContent = result.prediction.toLocaleString("id-ID", { maximumFractionDigits: 2 });
    const gram = await convertToGram(result.prediction);
    gramResult.textContent = gram.price_per_gram.toLocaleString("id-ID", { maximumFractionDigits: 2 });
    predictionPeriod.textContent = `${months[result.month - 1]} ${result.year} · period ${result.period}`;
  } catch (error) {
    predictionResult.textContent = "-";
    gramResult.textContent = "-";
    predictionError.textContent = error.message;
    predictionError.hidden = false;
  }
});

renderPeriodSelectors();
monthSlider.addEventListener("input", () => {
  selectedMonth = Number(monthSlider.value);
  updateSelection();
});
