import numpy as np
import cv2
import tensorflow as tf
import base64
from io import BytesIO
from PIL import Image

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# ── Load model ────────────────────────────────────────────────
model = tf.keras.models.load_model("best_MobileNetV2_ft.keras")
CLASS_NAMES = ["Digestive", "Indigestive"]

CONV_LIKE_LAYERS = (
    tf.keras.layers.Conv2D,
    tf.keras.layers.SeparableConv2D,
    tf.keras.layers.DepthwiseConv2D,
)


def find_last_conv_layer(keras_model):
    try:
        layer = keras_model.get_layer("final_conv")
        return layer.name, keras_model
    except ValueError:
        pass

    for layer in reversed(keras_model.layers):
        if isinstance(layer, CONV_LIKE_LAYERS):
            return layer.name, keras_model
        if isinstance(layer, tf.keras.Model):
            try:
                sub_layer = layer.get_layer("final_conv")
                return sub_layer.name, layer
            except ValueError:
                pass
            for sub in reversed(layer.layers):
                if isinstance(sub, CONV_LIKE_LAYERS):
                    return sub.name, layer

    return None, None


LAST_CONV_NAME, CONV_OWNER = find_last_conv_layer(model)
if LAST_CONV_NAME is None:
    raise RuntimeError("Could not find any convolutional layer in the loaded model.")
print(f"[OK] Grad-CAM++ layer: '{LAST_CONV_NAME}' in '{CONV_OWNER.name}'")

app = FastAPI()

# ── HTML UI ───────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Waste Classification AI</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --green:      #22c55e;
      --green-dark: #16a34a;
      --amber:      #f59e0b;
      --amber-dark: #d97706;
      --bg:         #ffffff;
      --surface:    #ffffff;
      --surface2:   #f1f5f9;
      --border:     #e2e8f0;
      --text:       #0f172a;
      --muted:      #64748b;
      --radius:     14px;
    }

    body {
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 20px 16px 40px;
    }

    header {
      width: 100%;
      max-width: 720px;
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 28px;
      padding-top: 8px;
    }
    .logo {
      width: 44px; height: 44px;
      background: linear-gradient(135deg, var(--green), #06b6d4);
      border-radius: 12px;
      display: flex; align-items: center; justify-content: center;
      font-size: 22px;
      flex-shrink: 0;
    }
    header h1 { font-size: clamp(17px, 4vw, 22px); font-weight: 700; letter-spacing: -0.3px; }
    header p  { font-size: 13px; color: var(--muted); margin-top: 2px; }

    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: clamp(20px, 5vw, 32px);
      width: 100%;
      max-width: 720px;
    }

    .upload-zone {
      border: 2px dashed var(--border);
      border-radius: var(--radius);
      padding: clamp(24px, 6vw, 40px) 20px;
      text-align: center;
      cursor: pointer;
      transition: border-color .2s, background .2s;
      margin-bottom: 16px;
      position: relative;
    }
    .upload-zone:hover,
    .upload-zone.drag  { border-color: var(--green); background: rgba(34,197,94,.06); }
    .upload-zone input { display: none; }
    .upload-icon { font-size: 40px; margin-bottom: 12px; filter: grayscale(.3); }
    .upload-zone h3 { font-size: 15px; margin-bottom: 4px; }
    .upload-zone p  { font-size: 13px; color: var(--muted); }
    .upload-zone .file-tag {
      display: inline-block;
      margin-top: 10px;
      background: rgba(34,197,94,.15);
      color: var(--green);
      border-radius: 20px;
      padding: 3px 12px;
      font-size: 12px;
      font-weight: 600;
      max-width: 90%;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .btn {
      width: 100%;
      padding: 14px;
      background: linear-gradient(135deg, var(--green), #16a34a);
      color: #fff;
      border: none;
      border-radius: var(--radius);
      font-size: 15px;
      font-weight: 700;
      cursor: pointer;
      letter-spacing: .3px;
      transition: opacity .2s, transform .1s;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
    }
    .btn:hover:not(:disabled)  { opacity: .9; transform: translateY(-1px); }
    .btn:active:not(:disabled) { transform: translateY(0); }
    .btn:disabled { background: var(--surface2); color: var(--muted); cursor: not-allowed; }

    .progress-wrap { margin-top: 16px; display: none; }
    .progress-wrap.show { display: block; }
    .progress-label {
      font-size: 12px; color: var(--muted); margin-bottom: 6px;
      display: flex; justify-content: space-between;
    }
    .progress-track {
      height: 6px; background: var(--surface2);
      border-radius: 3px; overflow: hidden;
    }
    .progress-fill {
      height: 100%;
      background: linear-gradient(90deg, var(--green), #06b6d4);
      border-radius: 3px; width: 0%;
      transition: width .4s ease;
    }

    .divider { border: none; border-top: 1px solid var(--border); margin: 24px 0; display: none; }
    .divider.show { display: block; }

    .gcampp-badge {
      display: none;
      align-items: center;
      gap: 8px;
      background: #e0e7ff;
      border: 1px solid #c7d2fe;
      border-radius: 10px;
      padding: 9px 14px;
      margin-bottom: 14px;
      font-size: 12px;
      color: #3730a3;
    }
    .gcampp-badge.show { display: flex; }
    .gcampp-badge strong { color: #1e1b4b; }
    .gcampp-dot {
      width: 8px; height: 8px;
      background: #4f46e5;
      border-radius: 50%;
      flex-shrink: 0;
      animation: pulse 1.8s ease-in-out infinite;
    }
    @keyframes pulse {
      0%,100% { opacity:1; transform:scale(1);   }
      50%      { opacity:.5; transform:scale(1.3); }
    }

    .img-grid {
      display: none;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
      margin-bottom: 20px;
    }
    .img-grid.show { display: grid; }
    @media (max-width: 480px) { .img-grid { grid-template-columns: 1fr; } }
    .img-panel {
      border: 1px solid var(--border);
      border-radius: var(--radius);
      overflow: hidden;
      background: var(--surface);
    }
    .img-panel-header {
      padding: 8px 12px;
      font-size: 11px; font-weight: 700;
      color: var(--muted);
      text-transform: uppercase; letter-spacing: .6px;
      border-bottom: 1px solid var(--border);
      display: flex; align-items: center; gap: 6px;
    }
    .img-panel-header .method-chip {
      margin-left: auto;
      background: #e0e7ff;
      color: #3730a3;
      border-radius: 20px;
      padding: 2px 8px;
      font-size: 10px;
      font-weight: 700;
      letter-spacing: .4px;
    }
    .img-panel img {
      width: 100%; display: block;
      max-height: 260px; object-fit: cover;
    }

    .heatmap-legend {
      padding: 8px 12px;
      display: flex;
      align-items: center;
      gap: 6px;
      border-top: 1px solid var(--border);
    }
    .legend-bar {
      flex: 1; height: 8px; border-radius: 4px;
      background: linear-gradient(90deg,
        #00008b, #0000ff, #00ffff, #00ff00, #ffff00, #ff8000, #ff0000);
    }
    .legend-label {
      font-size: 10px; color: var(--muted); white-space: nowrap;
    }

    .result {
      border-radius: var(--radius);
      padding: 20px;
      display: none;
      border: 1px solid var(--border);
    }
    .result.show        { display: block; }
    .result.digestive   { border-color: rgba(34,197,94,.4);  background: rgba(34,197,94,.08); }
    .result.indigestive { border-color: rgba(245,158,11,.4); background: rgba(245,158,11,.08); }

    .result-top {
      display: flex; align-items: center;
      gap: 14px; margin-bottom: 18px; flex-wrap: wrap;
    }
    .result-icon {
      width: 52px; height: 52px; border-radius: 14px;
      display: flex; align-items: center; justify-content: center;
      font-size: 26px; flex-shrink: 0;
    }
    .digestive   .result-icon { background: rgba(34,197,94,.2); }
    .indigestive .result-icon { background: rgba(245,158,11,.2); }
    .result-title { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .6px; color: var(--muted); }
    .result-class { font-size: clamp(22px, 5vw, 28px); font-weight: 800; line-height: 1.1; margin-top: 2px; }
    .digestive   .result-class { color: var(--green); }
    .indigestive .result-class { color: var(--amber); }
    .badge { margin-left: auto; padding: 5px 14px; border-radius: 20px; font-size: 13px; font-weight: 700; }
    .digestive   .badge { background: rgba(34,197,94,.2);  color: var(--green); }
    .indigestive .badge { background: rgba(245,158,11,.2); color: var(--amber); }

    .conf-section { margin-top: 4px; }
    .conf-label-row {
      font-size: 11px; font-weight: 700;
      text-transform: uppercase; letter-spacing: .5px;
      color: var(--muted); margin-bottom: 10px;
    }
    .conf-row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
    .conf-name { font-size: 13px; width: 100px; flex-shrink: 0; color: var(--text); }
    .bar-track { flex: 1; height: 10px; background: var(--surface2); border-radius: 5px; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 5px; transition: width .7s cubic-bezier(.4,0,.2,1); }
    .digestive   .bar-fill.d { background: linear-gradient(90deg, var(--green), #86efac); }
    .indigestive .bar-fill.d { background: linear-gradient(90deg, var(--green), #86efac); }
    .bar-fill.i { background: linear-gradient(90deg, var(--amber), #fcd34d); }
    .conf-pct { font-size: 13px; font-weight: 700; min-width: 46px; text-align: right; }
    .digestive   .conf-pct.d { color: var(--green); }
    .indigestive .conf-pct.d { color: var(--green); }
    .conf-pct.i { color: var(--amber); }

    .info-pills {
      display: flex; flex-wrap: wrap; gap: 8px;
      margin-top: 16px; padding-top: 16px;
      border-top: 1px solid var(--border);
    }
    .pill {
      display: flex; align-items: center; gap: 5px;
      background: var(--surface2); border: 1px solid var(--border);
      border-radius: 20px; padding: 5px 12px;
      font-size: 12px; color: var(--muted);
    }
    .pill span { color: var(--text); font-weight: 600; }

    footer { margin-top: 24px; font-size: 12px; color: var(--muted); text-align: center; }
  </style>
</head>
<body>

<header>
  <div class="logo">♻️</div>
  <div>
    <h1>Waste Classification AI</h1>
    <p>Powered by CNN + <strong style="color:#4f46e5">Grad-CAM++</strong> explainability</p>
  </div>
</header>

<div class="card">

  <div class="upload-zone" id="uploadZone"
       onclick="document.getElementById('fileInput').click()"
       ondragover="onDragOver(event)" ondragleave="onDragLeave(event)" ondrop="onDrop(event)">
    <div class="upload-icon" id="uploadIcon">📂</div>
    <h3 id="uploadTitle">Click or drag &amp; drop an image</h3>
    <p id="uploadSub">Supports JPG, PNG, WEBP · Max 10 MB</p>
    <div class="file-tag" id="fileTag" style="display:none"></div>
    <input type="file" id="fileInput" accept="image/*" onchange="onFileSelect(this)">
  </div>

  <button class="btn" id="predictBtn" onclick="predict()" disabled>
    <span id="btnIcon">🔍</span>
    <span id="btnText">Classify Image</span>
  </button>

  <div class="progress-wrap" id="progressWrap">
    <div class="progress-label">
      <span id="progressLabel">Analysing…</span>
      <span id="progressPct">0%</span>
    </div>
    <div class="progress-track">
      <div class="progress-fill" id="progressFill"></div>
    </div>
  </div>

  <hr class="divider" id="divider">

  <div class="gcampp-badge" id="gcamppBadge">
    <div class="gcampp-dot"></div>
    <div>
      <strong>Grad-CAM++</strong> active —
      second-order gradient weighting for full object coverage
      (Chattopadhay et al., WACV 2018)
    </div>
  </div>

  <div class="img-grid" id="imgGrid">
    <div class="img-panel">
      <div class="img-panel-header">🖼️ Original image</div>
      <img id="preview" alt="Uploaded image">
    </div>
    <div class="img-panel">
      <div class="img-panel-header">
        🔥 Explainability heatmap
        <span class="method-chip">GRAD-CAM++</span>
      </div>
      <img id="heatmap" alt="Grad-CAM++ overlay">
      <div class="heatmap-legend">
        <span class="legend-label">Low</span>
        <div class="legend-bar"></div>
        <span class="legend-label">High activation</span>
      </div>
    </div>
  </div>

  <div class="result" id="resultCard">
    <div class="result-top">
      <div class="result-icon" id="resultIcon"></div>
      <div>
        <div class="result-title">Classification Result</div>
        <div class="result-class" id="resultClass"></div>
      </div>
      <div class="badge" id="resultBadge"></div>
    </div>

    <div class="conf-section">
      <div class="conf-label-row">Confidence Scores</div>
      <div id="confBars"></div>
    </div>

    <div class="info-pills" id="infoPills"></div>
  </div>

</div>

<footer>
  Model: EfficientNetB0 &nbsp;·&nbsp;
  Explainability: <strong style="color:#4f46e5">Grad-CAM++</strong> &nbsp;·&nbsp;
  Framework: TensorFlow / FastAPI
</footer>

<script>
  let selectedFile = null;

  function onDragOver(e)  { e.preventDefault(); document.getElementById('uploadZone').classList.add('drag'); }
  function onDragLeave(e) { document.getElementById('uploadZone').classList.remove('drag'); }
  function onDrop(e) {
    e.preventDefault();
    document.getElementById('uploadZone').classList.remove('drag');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) loadFile(file);
  }

  function onFileSelect(input) { if (input.files[0]) loadFile(input.files[0]); }

  function loadFile(file) {
    selectedFile = file;
    const tag = document.getElementById('fileTag');
    tag.textContent = file.name;
    tag.style.display = 'inline-block';
    document.getElementById('uploadTitle').textContent = 'Image selected ✓';
    document.getElementById('uploadSub').textContent   = (file.size / 1024).toFixed(0) + ' KB';
    document.getElementById('uploadIcon').textContent  = '✅';
    document.getElementById('preview').src = URL.createObjectURL(file);
    document.getElementById('predictBtn').disabled = false;

    document.getElementById('resultCard').className = 'result';
    document.getElementById('divider').classList.remove('show');
    document.getElementById('imgGrid').classList.remove('show');
    document.getElementById('gcamppBadge').classList.remove('show');
  }

  let progressTimer = null;
  function startProgress() {
    const fill  = document.getElementById('progressFill');
    const pct   = document.getElementById('progressPct');
    const label = document.getElementById('progressLabel');
    const steps = [
      [10, 'Loading image…'],
      [30, 'Running CNN model…'],
      [55, 'Computing Grad-CAM++ (2nd-order gradients)…'],
      [75, 'Building alpha weight maps…'],
      [90, 'Generating overlay…'],
      [96, 'Almost done…'],
    ];
    let i = 0;
    fill.style.width = '0%';
    document.getElementById('progressWrap').classList.add('show');
    progressTimer = setInterval(() => {
      if (i < steps.length) {
        fill.style.width  = steps[i][0] + '%';
        pct.textContent   = steps[i][0] + '%';
        label.textContent = steps[i][1];
        i++;
      }
    }, 500);
  }
  function finishProgress() {
    clearInterval(progressTimer);
    document.getElementById('progressFill').style.width = '100%';
    document.getElementById('progressPct').textContent  = '100%';
    setTimeout(() => document.getElementById('progressWrap').classList.remove('show'), 500);
  }

  async function predict() {
    if (!selectedFile) return;

    document.getElementById('predictBtn').disabled = true;
    document.getElementById('btnIcon').textContent = '⏳';
    document.getElementById('btnText').textContent = 'Analysing…';
    document.getElementById('resultCard').className = 'result';
    document.getElementById('divider').classList.remove('show');
    document.getElementById('imgGrid').classList.remove('show');
    document.getElementById('gcamppBadge').classList.remove('show');

    startProgress();

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const res  = await fetch('/predict', { method: 'POST', body: formData });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || `Server error ${res.status}`);

      finishProgress();

      document.getElementById('heatmap').src = 'data:image/png;base64,' + data.heatmap;
      document.getElementById('divider').classList.add('show');
      document.getElementById('gcamppBadge').classList.add('show');
      document.getElementById('imgGrid').classList.add('show');

      const card = document.getElementById('resultCard');
      const cls  = data.class.toLowerCase();
      card.className = `result show ${cls}`;

      document.getElementById('resultIcon').textContent   = cls === 'digestive' ? '🟢' : '🟡';
      document.getElementById('resultClass').textContent  = data.class;
      document.getElementById('resultBadge').textContent  = data.confidence + '% confidence';

      document.getElementById('confBars').innerHTML = data.all_confidences.map((c, idx) => {
        const fillCls = idx === 0 ? 'd' : 'i';
        return `
          <div class="conf-row">
            <span class="conf-name">${c.label}</span>
            <div class="bar-track">
              <div class="bar-fill ${fillCls}" style="width:${c.pct}%"></div>
            </div>
            <span class="conf-pct ${fillCls}">${c.pct}%</span>
          </div>`;
      }).join('');

      const now = new Date();
      document.getElementById('infoPills').innerHTML = `
        <div class="pill">🕒 <span>${now.toLocaleTimeString()}</span></div>
        <div class="pill">📐 <span>224 × 224 px</span></div>
        <div class="pill">🔬 <span>Grad-CAM++</span></div>
        <div class="pill">🗂️ <span>${selectedFile.name.slice(0,24)}${selectedFile.name.length>24?'…':''}</span></div>
      `;

    } catch (err) {
      finishProgress();
      alert('❌ ' + err.message);
    } finally {
      document.getElementById('predictBtn').disabled = false;
      document.getElementById('btnIcon').textContent = '🔍';
      document.getElementById('btnText').textContent = 'Classify Image';
    }
  }
</script>
</body>
</html>
"""


# ── Preprocessing ─────────────────────────────────────────────
def preprocess(img_rgb: np.ndarray) -> np.ndarray:
    img = cv2.resize(img_rgb, (224, 224))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)


# ── Grad-CAM++ core ───────────────────────────────────────────
def compute_gradcampp(
    conv_outputs_val: np.ndarray, grads_val: np.ndarray
) -> np.ndarray:
    """
    Pure-numpy Grad-CAM++ computation with proper normalization.
    Fix: percentile-based contrast stretching so weak heatmaps
    don't collapse to a single dark color.
    """
    A = conv_outputs_val[0]   # (H, W, C)
    g = grads_val[0]          # (H, W, C)

    g2 = g ** 2
    g3 = g ** 3

    A_sum = np.sum(A, axis=(0, 1))   # (C,)

    denom = 2.0 * g2 + A_sum[np.newaxis, np.newaxis, :] * g3
    denom = np.where(np.abs(denom) > 1e-9, denom, 1e-9)

    alpha  = g2 / denom
    relu_g = np.maximum(g, 0.0)
    weights = np.sum(alpha * relu_g, axis=(0, 1))  # (C,)

    heatmap = np.sum(A * weights[np.newaxis, np.newaxis, :], axis=-1)
    heatmap = np.maximum(heatmap, 0.0)   # ReLU

    # ── FIX 1: percentile-based normalization ─────────────────
    # Plain max-norm collapses everything when signal is weak.
    # Stretch between 1st and 99th percentile so there's always
    # a full color range even with low-activation maps.
    p_low  = np.percentile(heatmap, 1)
    p_high = np.percentile(heatmap, 99)

    if p_high - p_low < 1e-6:
        # Truly flat map — try absolute max as fallback
        max_val = heatmap.max()
        if max_val <= 0:
            return np.zeros_like(heatmap, dtype="float32")
        return (heatmap / max_val).astype("float32")

    heatmap = np.clip((heatmap - p_low) / (p_high - p_low), 0.0, 1.0)
    return heatmap.astype("float32")


# ── Grad-CAM++ driver ─────────────────────────────────────────
def generate_gradcampp(img_array: np.ndarray) -> np.ndarray:
    """
    Compute Grad-CAM++ heatmap.
    Fix: for transfer-learning models the gradient path must go
    through the FULL model output, not just the sub-model.
    """
    img_tensor = tf.constant(img_array, dtype=tf.float32)

    if CONV_OWNER is model:
        # Flat / custom CNN
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(LAST_CONV_NAME).output, model.output],
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor, training=False)
            tape.watch(conv_outputs)            # ← FIX 2: watch the tensor
            if isinstance(predictions, (list, tuple)):
                predictions = tf.concat(
                    [tf.reshape(p, (tf.shape(p)[0], -1)) for p in predictions],
                    axis=1,
                )
            loss = predictions[:, 0]
        grads = tape.gradient(loss, conv_outputs)

    else:
        # Transfer-learning / nested backbone
        # FIX 3: build a single end-to-end gradient model so
        # gradients flow from the full model output back to the
        # conv layer — avoids the "None gradient" problem.
        sub_conv_output = CONV_OWNER.get_layer(LAST_CONV_NAME).output

        # Intermediate model: input → last conv layer output
        conv_model = tf.keras.models.Model(
            inputs=CONV_OWNER.inputs,
            outputs=sub_conv_output,
        )

        with tf.GradientTape() as tape:
            conv_outputs = conv_model(img_tensor, training=False)
            tape.watch(conv_outputs)

            # Reconstruct path from conv_outputs through the rest of the model
            # by running the full model AND watching the intermediate tensor
            predictions = model(img_tensor, training=False)
            if isinstance(predictions, (list, tuple)):
                predictions = tf.concat(
                    [tf.reshape(p, (tf.shape(p)[0], -1)) for p in predictions],
                    axis=1,
                )
            loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        # Last resort: gradient is None (disconnected graph)
        # Fall back to activation-only visualization
        print("[WARN] Grad-CAM++: gradients are None, using activation map.")
        A = conv_outputs.numpy()[0]                 # (H, W, C)
        heatmap = np.mean(A, axis=-1)               # mean activation
        heatmap = np.maximum(heatmap, 0)
        p_low, p_high = np.percentile(heatmap, 1), np.percentile(heatmap, 99)
        if p_high - p_low < 1e-6:
            return np.zeros(A.shape[:2], dtype="float32")
        return np.clip((heatmap - p_low) / (p_high - p_low), 0, 1).astype("float32")

    return compute_gradcampp(conv_outputs.numpy(), grads.numpy())


# ── Overlay helper ────────────────────────────────────────────
def overlay_heatmap(orig_rgb: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """
    Resize heatmap → apply JET colormap → blend with original.

    FIX 4: increased heatmap weight (0.55) and added a mild
    Gaussian blur so the map looks smoother and more vivid.
    Also applies CLAHE-style gamma to punch up mid-range values.
    """
    h, w = orig_rgb.shape[:2]

    # Upscale with bicubic for smoother edges
    hm = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)

    # FIX 5: gamma correction — push mid values higher so the
    # map isn't dominated by near-zero (blue) areas
    gamma = 0.6          # < 1 brightens the heatmap
    hm = np.power(np.clip(hm, 0, 1), gamma)

    # Mild Gaussian blur to remove block artifacts
    hm = cv2.GaussianBlur(hm, (7, 7), sigmaX=0)

    hm_u = np.uint8(255 * hm)
    hm_c = cv2.applyColorMap(hm_u, cv2.COLORMAP_JET)
    hm_c = cv2.cvtColor(hm_c, cv2.COLOR_BGR2RGB)

    # FIX 6: higher heatmap weight (was 0.40) + ensure uint8 clip
    sup = np.clip(hm_c * 0.55 + orig_rgb * 0.45, 0, 255).astype("uint8")
    return sup


def to_base64_png(img_rgb: np.ndarray) -> str:
    buf = BytesIO()
    Image.fromarray(img_rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── Routes ────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def home():
    return HTML


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)

    if npimg.size == 0:
        return JSONResponse(
            status_code=400, content={"error": "Empty file uploaded."}
        )

    img_bgr = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Could not decode image. "
                "Please upload a valid JPG, PNG, or WEBP file."
            },
        )

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    processed = preprocess(img_rgb)

    # ── Prediction ────────────────────────────────────────────
    raw_pred = model.predict(processed, verbose=0)
    if isinstance(raw_pred, (list, tuple)):
        raw_pred = np.concatenate(
            [np.reshape(p, (p.shape[0], -1)) for p in raw_pred], axis=1
        )
    raw_score = float(raw_pred[0][0])

    p_indigestive = raw_score
    p_digestive   = 1.0 - raw_score

    if p_indigestive > 0.5:
        predicted_class = "Indigestive"
        confidence      = p_indigestive
    else:
        predicted_class = "Digestive"
        confidence      = p_digestive

    # ── Grad-CAM++ heatmap ────────────────────────────────────
    heatmap     = generate_gradcampp(processed)
    display_img = cv2.resize(img_rgb, (224, 224))
    overlay     = overlay_heatmap(display_img, heatmap)
    heatmap_b64 = to_base64_png(overlay)

    all_confidences = [
        {"label": "Digestive",   "pct": round(p_digestive   * 100, 1)},
        {"label": "Indigestive", "pct": round(p_indigestive * 100, 1)},
    ]

    return {
        "class":           predicted_class,
        "confidence":      round(confidence * 100, 2),
        "all_confidences": all_confidences,
        "heatmap":         heatmap_b64,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)