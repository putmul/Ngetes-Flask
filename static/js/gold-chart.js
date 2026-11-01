const goldChart = document.getElementById("gold-chart");
const goldChartError = document.getElementById("gold-chart-error");

async function loadGoldHistory() {
  try {
    const response = await fetch("/api/gold-history");
    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.error || "Data harga emas tidak dapat dimuat.");
    }

    Plotly.react(
      goldChart,
      [
        {
          x: result.dates,
          y: result.prices,
          mode: "lines",
          name: "Harga emas",
          line: { color: "#126b63", width: 2 },
          hovertemplate: "Tanggal: %{x}<br>Harga: %{y:,.2f}<extra></extra>",
        },
      ],
      {
        margin: { t: 10, r: 20, b: 55, l: 60 },
        hovermode: "x unified",
        xaxis: {
          title: "Periode",
          type: "date",
        },
        yaxis: { title: "Harga" },
      },
      { responsive: true, displaylogo: false }
    );
  } catch (error) {
    goldChartError.textContent = error.message;
    goldChartError.hidden = false;
  }
}

loadGoldHistory();
