const clusterCount = document.getElementById("cluster-count");
const clusterError = document.getElementById("clustering-error");

function groupPoints(points) {
  const groups = {};
  points.forEach((point) => {
    const key = `Cluster ${point.cluster}`;
    if (!groups[key]) groups[key] = { x: [], y: [], text: [], name: key, mode: "markers", type: "scattergl", marker: { size: 8 } };
    groups[key].x.push(point.x);
    groups[key].y.push(point.y);
    groups[key].text.push(`ID ${point.id}<br>${point.sentiment}<br>${point.text}`);
  });
  return Object.values(groups);
}

function renderSummary(result) {
  const summary = document.getElementById("cluster-summary");
  summary.textContent = "";
  Object.entries(result.cluster_counts).forEach(([cluster, count]) => {
    const item = document.createElement("div");
    item.className = "cluster-item";
    const title = document.createElement("strong");
    title.textContent = `Cluster ${cluster}: ${count} data`;
    const keywords = document.createElement("small");
    keywords.textContent = `Kata utama: ${(result.keywords[cluster] || []).join(", ")}`;
    item.append(title, keywords);
    summary.appendChild(item);
  });
}

async function loadClusters() {
  clusterError.hidden = true;
  const k = clusterCount.value;
  try {
    const response = await fetch(`/api/clusters?k=${encodeURIComponent(k)}`);
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Clustering gagal.");
    document.getElementById("silhouette-score").textContent = result.silhouette_score ?? "-";
    renderSummary(result);
    Plotly.react("cluster-chart", groupPoints(result.points), {
      margin: { t: 10, r: 10, b: 45, l: 45 },
      xaxis: { title: "Komponen 1" }, yaxis: { title: "Komponen 2" },
      hovermode: "closest", legend: { orientation: "h" },
    }, { responsive: true, displaylogo: false });
  } catch (error) {
    clusterError.textContent = error.message;
    clusterError.hidden = false;
  }
}

document.getElementById("load-clusters").addEventListener("click", loadClusters);
loadClusters();
