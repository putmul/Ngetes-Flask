const dashboardChart = document.getElementById("sentiment-chart");

async function loadDashboard() {
  try {
    const [summaryResponse, distributionResponse] = await Promise.all([
      fetch("/api/summary"),
      fetch("/api/sentiment-distribution"),
    ]);
    if (!summaryResponse.ok || !distributionResponse.ok) throw new Error("Data dashboard tidak dapat dimuat.");
    const summary = await summaryResponse.json();
    const distribution = await distributionResponse.json();
    const counts = summary.sentiment_counts;
    document.getElementById("total-records").textContent = summary.total_records;
    document.getElementById("positive-count").textContent = counts.positive || 0;
    document.getElementById("negative-count").textContent = counts.negative || 0;
    document.getElementById("neutral-count").textContent = counts.neutral || 0;
    document.getElementById("average-length").textContent = `${summary.average_text_length} karakter`;
    document.getElementById("average-words").textContent = `${summary.average_word_count} kata`;
    document.getElementById("duplicate-texts").textContent = summary.duplicate_texts;
    new Chart(dashboardChart, {
      type: "doughnut",
      data: { labels: distribution.labels, datasets: [{ data: distribution.values, backgroundColor: ["#1b8b76", "#dc6b5b", "#f0b44d"] }] },
      options: { maintainAspectRatio: false, plugins: { legend: { position: "bottom" } } },
    });
  } catch (error) {
    const element = document.getElementById("dashboard-error");
    element.textContent = error.message;
    element.hidden = false;
  }
}

loadDashboard();
