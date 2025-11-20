// app.js - minimal JS, clean UI
const video = document.getElementById('video');
const thumb = document.getElementById('thumb');
const predStatus = document.getElementById('predStatus');
const predDetails = document.getElementById('predDetails');
const logsDiv = document.getElementById('logs');
const totalsDiv = document.getElementById('totals');

async function startCamera(){
  try {
    const s = await navigator.mediaDevices.getUserMedia({video:true});
    video.srcObject = s;
  } catch(e){
    alert('Camera not available: ' + e);
  }
}
startCamera();

document.getElementById('capture').addEventListener('click', async ()=>{
  const c = document.createElement('canvas'); c.width=224; c.height=224;
  c.getContext('2d').drawImage(video,0,0,c.width,c.height);
  const dataUrl = c.toDataURL('image/jpeg',0.9);
  thumb.src = dataUrl;
  const serving = Number(document.getElementById('serv').value) || 100;

  predStatus.textContent = 'Predicting...';
  predDetails.innerHTML = '';
  try {
    const res = await fetch('http://localhost:5000/predict', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        image_base64: dataUrl,
        serving_g: serving})
    });
    const j = await res.json();
    if (j.error){ predStatus.textContent = 'Error: ' + j.error; return; }
    showPrediction(j.predictions[0]);
  } catch(e){
    predStatus.textContent = 'Request failed: ' + e;
  }
});

function showPrediction(p){
  if (!p) { predStatus.textContent = 'No prediction'; return; }
  predStatus.textContent = `${p.label} — ${(p.confidence*100).toFixed(1)}%`;
  if (p.nutrition && p.nutrition.per_serving){
    const n = p.nutrition.per_serving;
    predDetails.innerHTML = `
      <div>Serving ${p.nutrition.serving_g} g</div>
      <div style="margin-top:6px;">
        <strong>${n.calories_kcal} kcal</strong> • ${n.protein_g||0} g protein • ${n.carbs_g||0} g carbs • ${n.fat_g||0} g fat
      </div>
      <button id="logBtn" class="btn primary" style="margin-top:8px">Log meal</button>
    `;
    document.getElementById('logBtn').onclick = ()=> addLog(p);
  } else {
    predDetails.innerHTML = '<div class="muted">No nutrition data</div>';
  }
}

function addLog(p){
  const logs = JSON.parse(localStorage.getItem('logs')||'[]');
  const entry = {
    label: p.label,
    time: new Date().toISOString(),
    serving_g: p.nutrition ? p.nutrition.serving_g : Number(document.getElementById('serv').value)||100,
    nutrition: p.nutrition ? p.nutrition.per_serving : null
  };
  logs.push(entry);
  localStorage.setItem('logs', JSON.stringify(logs));
  renderLogs();
}

function renderLogs(){
  const all = JSON.parse(localStorage.getItem('logs')||'[]');
  const today = new Date().toISOString().slice(0,10);
  const todays = all.filter(x => x.time.slice(0,10) === today).reverse();
  logsDiv.innerHTML = '';
  if (!todays.length){ logsDiv.innerHTML = '<div class="muted">No logs for today</div>'; totalsDiv.innerText=''; return; }
  let totals = {cal:0,protein:0,carb:0,fat:0};
  todays.forEach(e=>{
    const d = document.createElement('div'); d.className='log-item';
    d.innerHTML = `<div class="log-left">${e.label}<div class="muted">${new Date(e.time).toLocaleTimeString()} • ${e.serving_g} g</div></div>
                   <div class="log-right">${e.nutrition ? Math.round(e.nutrition.calories_kcal) + ' kcal' : '—'}</div>`;
    logsDiv.appendChild(d);
    if (e.nutrition){ totals.cal += e.nutrition.calories_kcal||0; totals.protein += e.nutrition.protein_g||0; totals.carb += e.nutrition.carbs_g||0; totals.fat += e.nutrition.fat_g||0; }
  });
  totalsDiv.innerText = `Totals today: ${Math.round(totals.cal)} kcal • Protein: ${(totals.protein||0).toFixed(1)} g`;
}

document.getElementById('clearAll').addEventListener('click', ()=>{
  if (!confirm('Clear all logs?')) return;
  localStorage.removeItem('logs');
  renderLogs();
});

renderLogs();
