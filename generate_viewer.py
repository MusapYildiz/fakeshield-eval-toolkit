#!/usr/bin/env python3
import os
"""
FakeShield Sonuç Görüntüleyici
Sunucuda çalıştır: python3 generate_viewer.py
Sonra tarayıcıda: http://localhost:9090/viewer.html
"""

import os, json, base64
from pathlib import Path

BASE_DIR    = os.environ.get("FAKESHIELD_DIR", "./FakeShield")
RESULTS_DIR = f"{BASE_DIR}/results"
SP_DIR      = f"{BASE_DIR}/datasets/casia_gt_test/Sp"
GT_DIR      = f"{BASE_DIR}/datasets/casia_gt_test/GT"
IOC_REPORT  = f"{RESULTS_DIR}/iou_report.json"
OUTPUT_HTML = f"{RESULTS_DIR}/viewer.html"


def img_to_base64(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        data = base64.b64encode(f.read()).decode()
    ext = Path(path).suffix.lower().replace('.', '')
    if ext == 'jpg': ext = 'jpeg'
    return f"data:image/{ext};base64,{data}"


def find_mflm_mask(stem):
    mflm_dir = f"{RESULTS_DIR}/{stem}_mflm"
    if not os.path.exists(mflm_dir):
        return None
    for f in os.listdir(mflm_dir):
        if f.endswith(('.jpg', '.png')):
            return f"{mflm_dir}/{f}"
    return None


def load_dte_output(stem):
    path = f"{RESULTS_DIR}/{stem}_dte.jsonl"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    if not lines:
        return None
    try:
        return json.loads(lines[-1]).get('outputs', '')
    except:
        return lines[-1]


# IoU raporu yükle
iou_data = {}
if os.path.exists(IOC_REPORT):
    with open(IOC_REPORT) as f:
        report = json.load(f)
    for s in report.get('per_sample', []):
        iou_data[s['image']] = s

# Sp görüntülerini işle
sp_images = sorted([f for f in os.listdir(SP_DIR) if f.endswith(('.jpg','.png'))])

cards = []
for img_name in sp_images:
    stem = img_name.replace('.jpg','').replace('.png','')
    
    orig_path = f"{SP_DIR}/{img_name}"
    gt_path   = f"{GT_DIR}/{stem}_gt.png"
    mask_path = find_mflm_mask(stem)
    dte_text  = load_dte_output(stem)
    
    iou_info  = iou_data.get(img_name, {})
    iou_val   = iou_info.get('iou')
    f1_val    = iou_info.get('f1')
    
    orig_b64 = img_to_base64(orig_path)
    gt_b64   = img_to_base64(gt_path)
    mask_b64 = img_to_base64(mask_path)
    
    if not orig_b64:
        continue
    
    iou_color = '#00c853' if iou_val and iou_val > 0.5 else \
                '#ff9800' if iou_val and iou_val > 0.3 else '#f44336'
    
    iou_str = f"{iou_val:.3f}" if iou_val is not None else "N/A"
    f1_str  = f"{f1_val:.3f}"  if f1_val  is not None else "N/A"
    
    cards.append({
        'name': stem,
        'orig': orig_b64,
        'gt':   gt_b64,
        'mask': mask_b64,
        'text': dte_text or 'DTE çıktısı yok',
        'iou':  iou_str,
        'f1':   f1_str,
        'iou_color': iou_color
    })

# HTML üret
html_cards = ""
for i, c in enumerate(cards):
    gt_html   = f'<img src="{c["gt"]}"   alt="GT Mask">'   if c['gt']   else '<div class="no-img">GT Maskesi Yok</div>'
    mask_html = f'<img src="{c["mask"]}" alt="MFLM Mask">' if c['mask'] else '<div class="no-img">MFLM Maskesi Yok</div>'
    
    text_escaped = c['text'].replace('<','&lt;').replace('>','&gt;').replace('\n','<br>')
    
    html_cards += f"""
    <div class="card" id="card-{i}">
        <div class="card-header">
            <span class="card-num">#{i+1}</span>
            <span class="card-name">{c['name']}</span>
            <div class="scores">
                <span class="score-badge" style="background:{c['iou_color']}">IoU: {c['iou']}</span>
                <span class="score-badge" style="background:{c['iou_color']}">F1: {c['f1']}</span>
            </div>
        </div>
        <div class="images">
            <div class="img-box">
                <div class="img-label">Orijinal</div>
                <img src="{c['orig']}" alt="Original">
            </div>
            <div class="img-box">
                <div class="img-label">GT Maskesi</div>
                {gt_html}
            </div>
            <div class="img-box">
                <div class="img-label">MFLM Maskesi</div>
                {mask_html}
            </div>
        </div>
        <div class="explanation">
            <div class="exp-label">DTE-FDM Açıklaması</div>
            <div class="exp-text">{text_escaped}</div>
        </div>
        <div class="rating">
            <div class="rating-label">Manuel Değerlendirme:</div>
            <div class="rating-buttons">
                <button class="btn-rate" onclick="rate({i}, 5)" title="Açıklama ve maske mükemmel">⭐⭐⭐⭐⭐ Mükemmel</button>
                <button class="btn-rate" onclick="rate({i}, 4)" title="İyi ama küçük hatalar var">⭐⭐⭐⭐ İyi</button>
                <button class="btn-rate" onclick="rate({i}, 3)" title="Kısmen doğru">⭐⭐⭐ Orta</button>
                <button class="btn-rate" onclick="rate({i}, 2)" title="Büyük ölçüde yanlış">⭐⭐ Zayıf</button>
                <button class="btn-rate" onclick="rate({i}, 1)" title="Tamamen yanlış">⭐ Çok Kötü</button>
            </div>
            <div class="notes-box">
                <input type="text" class="notes-input" id="notes-{i}" placeholder="Not ekle...">
            </div>
            <div class="rating-result" id="result-{i}"></div>
        </div>
    </div>
    """

html = f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FakeShield Manuel Değerlendirme</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  
  body {{
    font-family: 'Courier New', monospace;
    background: #0a0a0a;
    color: #e0e0e0;
    min-height: 100vh;
  }}
  
  header {{
    background: #111;
    border-bottom: 2px solid #333;
    padding: 20px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
  }}
  
  header h1 {{
    font-size: 1.2rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #fff;
  }}
  
  header h1 span {{ color: #ff4444; }}
  
  .stats {{
    display: flex;
    gap: 20px;
    font-size: 0.8rem;
    color: #888;
  }}
  
  .stat {{ 
    display: flex; 
    flex-direction: column; 
    align-items: center;
  }}
  
  .stat-val {{
    font-size: 1.2rem;
    font-weight: bold;
    color: #fff;
  }}

  .progress-bar {{
    height: 3px;
    background: #1a1a1a;
    width: 100%;
  }}
  
  .progress-fill {{
    height: 100%;
    background: linear-gradient(90deg, #ff4444, #ff8800);
    transition: width 0.3s ease;
    width: 0%;
  }}

  main {{
    max-width: 1400px;
    margin: 0 auto;
    padding: 32px 20px;
    display: flex;
    flex-direction: column;
    gap: 32px;
  }}

  .card {{
    background: #111;
    border: 1px solid #222;
    border-radius: 4px;
    overflow: hidden;
    transition: border-color 0.2s;
  }}
  
  .card:hover {{ border-color: #444; }}
  .card.rated-good  {{ border-left: 4px solid #00c853; }}
  .card.rated-mid   {{ border-left: 4px solid #ff9800; }}
  .card.rated-bad   {{ border-left: 4px solid #f44336; }}

  .card-header {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 20px;
    background: #161616;
    border-bottom: 1px solid #222;
  }}
  
  .card-num {{
    font-size: 0.75rem;
    color: #555;
    min-width: 30px;
  }}
  
  .card-name {{
    font-size: 0.82rem;
    color: #aaa;
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }}
  
  .scores {{ display: flex; gap: 8px; }}
  
  .score-badge {{
    padding: 3px 10px;
    border-radius: 2px;
    font-size: 0.75rem;
    font-weight: bold;
    color: #fff;
  }}

  .images {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1px;
    background: #222;
  }}
  
  .img-box {{
    background: #0d0d0d;
    padding: 12px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
  }}
  
  .img-label {{
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #555;
  }}
  
  .img-box img {{
    max-width: 100%;
    max-height: 280px;
    object-fit: contain;
    border-radius: 2px;
  }}
  
  .no-img {{
    width: 100%;
    height: 200px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #333;
    font-size: 0.8rem;
    border: 1px dashed #222;
  }}

  .explanation {{
    padding: 16px 20px;
    border-top: 1px solid #1a1a1a;
  }}
  
  .exp-label {{
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #555;
    margin-bottom: 8px;
  }}
  
  .exp-text {{
    font-size: 0.82rem;
    line-height: 1.7;
    color: #bbb;
    max-height: 150px;
    overflow-y: auto;
    padding-right: 8px;
  }}
  
  .exp-text::-webkit-scrollbar {{ width: 4px; }}
  .exp-text::-webkit-scrollbar-track {{ background: #111; }}
  .exp-text::-webkit-scrollbar-thumb {{ background: #333; }}

  .rating {{
    padding: 16px 20px;
    border-top: 1px solid #1a1a1a;
    background: #0d0d0d;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }}
  
  .rating-label {{
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #555;
  }}
  
  .rating-buttons {{ display: flex; gap: 8px; flex-wrap: wrap; }}
  
  .btn-rate {{
    padding: 6px 14px;
    background: #1a1a1a;
    border: 1px solid #333;
    color: #888;
    font-size: 0.75rem;
    cursor: pointer;
    border-radius: 2px;
    transition: all 0.15s;
    font-family: inherit;
  }}
  
  .btn-rate:hover {{
    background: #252525;
    color: #fff;
    border-color: #555;
  }}
  
  .btn-rate.selected {{
    background: #333;
    color: #fff;
    border-color: #fff;
  }}
  
  .notes-input {{
    width: 100%;
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    color: #ccc;
    padding: 8px 12px;
    font-size: 0.8rem;
    font-family: inherit;
    border-radius: 2px;
    outline: none;
  }}
  
  .notes-input:focus {{ border-color: #444; }}
  
  .rating-result {{
    font-size: 0.8rem;
    color: #666;
    min-height: 20px;
  }}

  .export-btn {{
    position: fixed;
    bottom: 24px;
    right: 24px;
    background: #ff4444;
    color: #fff;
    border: none;
    padding: 14px 24px;
    font-size: 0.85rem;
    font-family: inherit;
    letter-spacing: 2px;
    text-transform: uppercase;
    cursor: pointer;
    border-radius: 2px;
    box-shadow: 0 4px 20px rgba(255,68,68,0.3);
    transition: all 0.2s;
  }}
  
  .export-btn:hover {{
    background: #ff2222;
    box-shadow: 0 4px 30px rgba(255,68,68,0.5);
    transform: translateY(-2px);
  }}

  @media (max-width: 768px) {{
    .images {{ grid-template-columns: 1fr; }}
    .rating-buttons {{ flex-direction: column; }}
  }}
</style>
</head>
<body>

<header>
  <h1>Fake<span>Shield</span> — Manuel Değerlendirme</h1>
  <div class="stats">
    <div class="stat"><span class="stat-val" id="total-count">{len(cards)}</span><span>Toplam</span></div>
    <div class="stat"><span class="stat-val" id="rated-count">0</span><span>Değerlendirilen</span></div>
    <div class="stat"><span class="stat-val" id="avg-score">-</span><span>Ort. Puan</span></div>
  </div>
</header>
<div class="progress-bar"><div class="progress-fill" id="progress"></div></div>

<main>
{html_cards}
</main>

<button class="export-btn" onclick="exportResults()">📥 Sonuçları İndir</button>

<script>
const ratings = {{}};
const total = {len(cards)};

function rate(idx, score) {{
  const notes = document.getElementById('notes-' + idx).value;
  ratings[idx] = {{ score, notes, card: document.querySelector('#card-' + idx + ' .card-name').textContent }};
  
  // Butonları güncelle
  document.querySelectorAll('#card-' + idx + ' .btn-rate').forEach(b => b.classList.remove('selected'));
  event.target.classList.add('selected');
  
  // Kart rengini güncelle
  const card = document.getElementById('card-' + idx);
  card.className = 'card ' + (score >= 4 ? 'rated-good' : score >= 3 ? 'rated-mid' : 'rated-bad');
  
  // Sonuç metni
  const labels = {{5:'Mükemmel ✅', 4:'İyi ✅', 3:'Orta ⚠️', 2:'Zayıf ❌', 1:'Çok Kötü ❌'}};
  document.getElementById('result-' + idx).textContent = 'Puan: ' + score + '/5 — ' + labels[score];
  
  updateStats();
}}

function updateStats() {{
  const rated = Object.keys(ratings).length;
  const scores = Object.values(ratings).map(r => r.score);
  const avg = scores.length ? (scores.reduce((a,b) => a+b, 0) / scores.length).toFixed(1) : '-';
  
  document.getElementById('rated-count').textContent = rated;
  document.getElementById('avg-score').textContent = avg;
  document.getElementById('progress').style.width = (rated/total*100) + '%';
}}

function exportResults() {{
  const data = Object.entries(ratings).map(([idx, r]) => ({{
    idx: parseInt(idx)+1,
    image: r.card,
    score: r.score,
    notes: r.notes
  }}));
  
  const scores = data.map(d => d.score);
  const avg = scores.length ? (scores.reduce((a,b)=>a+b,0)/scores.length).toFixed(2) : 0;
  const dist = {{5:0,4:0,3:0,2:0,1:0}};
  scores.forEach(s => dist[s]++);
  
  const summary = {{
    total_rated: data.length,
    total_images: total,
    avg_score: parseFloat(avg),
    distribution: dist,
    per_sample: data
  }};
  
  const blob = new Blob([JSON.stringify(summary, null, 2)], {{type:'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'fakeshield_manual_eval.json';
  a.click();
}}
</script>
</body>
</html>"""

with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"✅ Viewer oluşturuldu: {OUTPUT_HTML}")
print(f"   {len(cards)} görüntü eklendi")
print(f"\nSunucuda şunu çalıştır:")
print(f"   python3 -m http.server 9090 --directory {RESULTS_DIR}")
print(f"\nTarayıcıda aç:")
print(f"   http://localhost:9090/viewer.html")
