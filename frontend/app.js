const API = ""; // same-origin, FastAPI serves this file too

let map, originMarker, destMarker, routeLine;
let origin = null;
let destination = null;
let lastThreadId = null;

function initMap(center) {
  map = L.map("map").setView(center, 14);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "&copy; OpenStreetMap contributors",
  }).addTo(map);

  setOrigin(center);

  map.on("click", (e) => {
    if (!origin || destination) {
      // reset cycle: first click after a destination exists starts over
      setOrigin([e.latlng.lat, e.latlng.lng]);
      destination = null;
      if (destMarker) { map.removeLayer(destMarker); destMarker = null; }
      if (routeLine) { map.removeLayer(routeLine); routeLine = null; }
    } else {
      setDestination([e.latlng.lat, e.latlng.lng]);
    }
  });
}

function setOrigin(latlng) {
  origin = { lat: latlng[0], lon: latlng[1] };
  if (originMarker) map.removeLayer(originMarker);
  originMarker = L.marker(latlng, { title: "Старт" }).addTo(map).bindPopup("Старт").openPopup();
}

function setDestination(latlng) {
  destination = { lat: latlng[0], lon: latlng[1] };
  if (destMarker) map.removeLayer(destMarker);
  destMarker = L.marker(latlng, { title: "Назначение", icon: L.icon({
    iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
    shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
  }) }).addTo(map).bindPopup("Назначение").openPopup();
}

function drawRoute(routeOptions) {
  if (routeLine) map.removeLayer(routeLine);
  if (!routeOptions || !routeOptions.length) return;
  const best = routeOptions[0];
  const coords = best.geometry.coordinates.map((c) => [c[1], c[0]]);
  routeLine = L.polyline(coords, { color: "#2c5f6f", weight: 5 }).addTo(map);
  routeLine.bindPopup(
    `${best.distance_m} м, ${best.duration_min} мин, тень: ${Math.round(best.shade_fraction * 100)}%`
  );
  map.fitBounds(routeLine.getBounds(), { padding: [20, 20] });
}

// Render immediately with a fallback center; geolocation (if the user grants
// it in time) just recenters afterwards. Never block the initial render on
// a permission prompt the user might not answer.
const FALLBACK_CENTER = [51.1694, 71.4491]; // Astana
initMap(FALLBACK_CENTER);

navigator.geolocation?.getCurrentPosition(
  (pos) => {
    const center = [pos.coords.latitude, pos.coords.longitude];
    map.setView(center, 14);
    setOrigin(center);
  },
  () => {}, // fallback center already showing, nothing to do
  { timeout: 4000 }
);

document.getElementById("noSpfYet").addEventListener("change", (e) => {
  document.getElementById("spfMinutesRow").style.display = e.target.checked ? "none" : "block";
});

document.getElementById("analyzeSkinBtn").addEventListener("click", async () => {
  const file = document.getElementById("skinPhoto").files[0];
  const out = document.getElementById("skinResult");
  if (!file) { out.textContent = "Выберите файл фото."; return; }
  out.textContent = "Анализирую...";
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(`${API}/api/analyze-skin`, { method: "POST", body: fd });
  const data = await res.json();
  if (data.phototype) {
    document.getElementById("phototype").value = data.phototype;
    out.textContent = `Определён фототип ${data.phototype} (уверенность ${Math.round((data.confidence||0)*100)}%). ${data.note||""}`;
  } else {
    out.textContent = "Не удалось определить фототип.";
  }
});

document.getElementById("analyzeLabelBtn").addEventListener("click", async () => {
  const file = document.getElementById("labelPhoto").files[0];
  const out = document.getElementById("labelResult");
  if (!file) { out.textContent = "Выберите файл фото этикетки."; return; }
  out.textContent = "Читаю этикетку...";
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(`${API}/api/analyze-label`, { method: "POST", body: fd });
  const data = await res.json();
  if (data.spf) {
    const warn = data.spf < 30 ? " ⚠️ Ниже рекомендованного SPF 30+." : " ✅ Соответствует рекомендации SPF 30+.";
    out.textContent = `SPF ${data.spf}, broad-spectrum: ${data.broad_spectrum}, срок: ${data.expiry_date || "не найден"}.${warn}`;
  } else {
    out.textContent = "Не удалось прочитать этикетку: " + (data.raw_text_snippet || "");
  }
});

function riskBadge(band) {
  const labels = { low: "Низкий", moderate: "Умеренный", high: "Высокий", very_high: "Очень высокий", extreme: "Экстремальный" };
  return `<span class="risk-badge risk-${band}">${labels[band] || band}</span>`;
}

function renderResult(data) {
  const result = document.getElementById("result");
  if (data.status === "needs_confirmation") {
    lastThreadId = data.thread_id;
    document.getElementById("confirmMessage").textContent = data.interrupt.message;
    document.getElementById("confirmModal").classList.remove("hidden");
    result.innerHTML = "Ожидание подтверждения...";
    return;
  }
  lastThreadId = null;
  const risk = data.risk || {};
  result.innerHTML = `
    ${riskBadge(risk.risk_band)}
    <p>${data.final_recommendation || ""}</p>
    <p class="hint">UV сейчас: ${data.uv_data?.current_uv_index ?? "?"} · Модель: ${data.model_used || "?"}</p>
  `;
  drawRoute(data.route_options);

  const traceList = document.getElementById("traceList");
  traceList.innerHTML = (data.trace || []).map((t) => `<li>${t}</li>`).join("");

  const sourcesList = document.getElementById("sourcesList");
  sourcesList.innerHTML = (data.rag_sources || [])
    .map((s) => `<li><b>${s.source}</b>: ${s.text}</li>`)
    .join("") || "<li>нет (вопрос не задан)</li>";
}

async function submitRecommend() {
  const result = document.getElementById("result");
  if (!origin) { result.textContent = "Сначала выберите точку старта на карте."; return; }
  result.textContent = "Считаю риск, UV, маршрут...";

  const noSpf = document.getElementById("noSpfYet").checked;
  const payload = {
    phototype: Number(document.getElementById("phototype").value),
    origin,
    destination: destination || null,
    minutes_outside: Number(document.getElementById("minutesOutside").value),
    minutes_since_last_spf: noSpf ? null : Number(document.getElementById("minutesSinceSpf").value),
    sweating_or_swimming: document.getElementById("sweating").checked,
    question: document.getElementById("question").value || null,
  };

  const res = await fetch(`${API}/api/recommend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  renderResult(data);
}

document.getElementById("recommendBtn").addEventListener("click", submitRecommend);

async function resumeConfirm(confirmed) {
  document.getElementById("confirmModal").classList.add("hidden");
  if (!lastThreadId) return;
  const res = await fetch(`${API}/api/resume`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ thread_id: lastThreadId, confirmed }),
  });
  const data = await res.json();
  renderResult(data);
}

document.getElementById("confirmYes").addEventListener("click", () => resumeConfirm(true));
document.getElementById("confirmNo").addEventListener("click", () => resumeConfirm(false));
