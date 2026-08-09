#!/usr/bin/env python3
"""
build.py — Generate index.html with app.py inlined for PyScript.

PyScript runs Python in the browser via Pyodide (WebAssembly). The Python
source is loaded via <py-script> tag with src attribute, OR inlined directly.

This script inlines app.py into the HTML to make it self-contained (no
separate file fetch needed — works on HF Spaces static SDK).

Usage: python3 build.py
Output: index.html
"""
from pathlib import Path
import html

HERE = Path(__file__).parent
APP_PY = HERE / "app.py"
INDEX_HTML = HERE / "index.html"

PLACEHOLDER = "___PYTHON_CODE_GOES_HERE___"

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LocInsight ML Engine</title>
    <meta name="description" content="Browser-based site selection scoring for MAP Active Adiperkasa (MAA) — Bali PoC. Runs entirely in your browser via Pyodide (WebAssembly).">

    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: #f8fafc;
            color: #0f172a;
            line-height: 1.5;
        }
        .header {
            background: linear-gradient(135deg, #7A0A1A 0%, #C8102E 100%);
            color: white;
            padding: 24px 32px;
        }
        .header h1 { font-size: 24px; font-weight: 700; margin-bottom: 4px; }
        .header p { font-size: 13px; opacity: 0.9; }
        .header .meta { font-size: 11px; opacity: 0.7; margin-top: 6px; }

        .container { max-width: 1100px; margin: 0 auto; padding: 24px 20px 60px; }

        .tabs {
            display: flex;
            gap: 4px;
            border-bottom: 2px solid #e5e7eb;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .tab {
            padding: 10px 18px;
            background: none;
            border: none;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            color: #6b7280;
            border-bottom: 2px solid transparent;
            margin-bottom: -2px;
            transition: all 0.15s;
        }
        .tab:hover { color: #C8102E; }
        .tab.active { color: #C8102E; border-bottom-color: #C8102E; }

        .panel { display: none; }
        .panel.active { display: block; }

        .card {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 16px;
        }
        .card h2 { font-size: 17px; margin-bottom: 8px; color: #7A0A1A; }
        .card p { font-size: 13px; color: #4b5563; margin-bottom: 14px; }

        .row { display: flex; gap: 12px; flex-wrap: wrap; align-items: flex-end; }
        .field { flex: 1; min-width: 140px; }
        .field label { display: block; font-size: 12px; font-weight: 600; color: #374151; margin-bottom: 4px; }
        .field input, .field select {
            width: 100%;
            padding: 8px 10px;
            border: 1px solid #d1d5db;
            border-radius: 6px;
            font-size: 14px;
        }
        .field .hint { font-size: 11px; color: #9ca3af; margin-top: 2px; }

        button.btn {
            background: #C8102E;
            color: white;
            border: none;
            padding: 9px 20px;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.15s;
        }
        button.btn:hover { background: #7A0A1A; }
        button.btn:disabled { background: #d1d5db; cursor: not-allowed; }

        pre.output {
            background: #1e293b;
            color: #e2e8f0;
            padding: 14px;
            border-radius: 6px;
            font-family: 'Menlo', 'Monaco', 'Courier New', monospace;
            font-size: 12px;
            overflow-x: auto;
            max-height: 500px;
            overflow-y: auto;
            white-space: pre-wrap;
            word-break: break-word;
        }

        #boot-loader {
            position: fixed; inset: 0;
            display: flex; flex-direction: column;
            align-items: center; justify-content: center;
            background: linear-gradient(135deg, #7A0A1A 0%, #C8102E 100%);
            color: white; z-index: 9999;
            transition: opacity 0.4s ease-out;
        }
        #boot-loader h1 { font-size: 26px; margin-bottom: 8px; }
        #boot-loader p { font-size: 13px; opacity: 0.9; max-width: 460px; text-align: center; padding: 0 20px; margin-bottom: 24px; }
        .spinner {
            width: 40px; height: 40px;
            border: 4px solid rgba(255,255,255,0.25);
            border-top-color: white;
            border-radius: 50%;
            animation: spin 0.9s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        #boot-status { margin-top: 16px; font-size: 12px; opacity: 0.85; min-height: 1.2em; }
        .boot-hidden { opacity: 0; pointer-events: none; }

        footer {
            text-align: center;
            padding: 16px;
            font-size: 11px;
            color: #9ca3af;
            border-top: 1px solid #e5e7eb;
            margin-top: 40px;
        }

        .badge {
            display: inline-block;
            padding: 2px 8px;
            background: #fef3c7;
            color: #92400e;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            margin-left: 6px;
        }
        .badge.green { background: #d1fae5; color: #065f46; }
    </style>

    <!-- PyScript (Pyodide-powered) from CDN -->
    <link rel="stylesheet" href="https://pyscript.net/releases/2025.5.1/core.css">
    <script type="module" src="https://pyscript.net/releases/2025.5.1/core.js"></script>
</head>
<body>
    <div id="boot-loader">
        <h1>LocInsight ML Engine</h1>
        <p>Browser-based site selection scoring for MAP Active Adiperkasa (MAA) — Bali PoC</p>
        <div class="spinner"></div>
        <div id="boot-status">Downloading Pyodide runtime (~10MB)...</div>
    </div>

    <div class="header">
        <h1>LocInsight ML Engine</h1>
        <p>Browser-based site selection scoring for MAP Active Adiperkasa (MAA) — Bali PoC</p>
        <p class="meta">Runs entirely in your browser via Pyodide (WebAssembly) · HF Spaces Static SDK · Free tier compatible</p>
    </div>

    <div class="container">
        <div class="tabs">
            <button class="tab active" onclick="showTab('health', this)">Health Check</button>
            <button class="tab" onclick="showTab('predict', this)">Predict Site Score</button>
            <button class="tab" onclick="showTab('blank', this)">Blank Spots</button>
            <button class="tab" onclick="showTab('train', this)">Train Model</button>
            <button class="tab" onclick="showTab('data', this)">Data Explorer</button>
            <button class="tab" onclick="showTab('about', this)">About</button>
        </div>

        <!-- Health Check -->
        <div id="tab-health" class="panel active">
            <div class="card">
                <h2>Service Status</h2>
                <p>Quick liveness + Supabase connectivity check. No auth required — uses Supabase publishable key with RLS.</p>
                <button class="btn" onclick="runHealth()">Run Health Check</button>
                <pre id="health-output" class="output" style="margin-top:14px;">Click the button above to check service status.</pre>
            </div>
        </div>

        <!-- Predict -->
        <div id="tab-predict" class="panel">
            <div class="card">
                <h2>Store Success Probability Score</h2>
                <p>Input a candidate site coordinate to get a 0-100% score based on competitor density, POI density, demographics, and mall proximity. If a GBR model has been trained in the "Train Model" tab, it will be used; otherwise a transparent weighted heuristic applies.</p>
                <div class="row">
                    <div class="field">
                        <label>Latitude</label>
                        <input type="number" id="pred-lat" value="-8.6705" step="0.0001">
                        <div class="hint">Bali centroid default</div>
                    </div>
                    <div class="field">
                        <label>Longitude</label>
                        <input type="number" id="pred-lng" value="115.2126" step="0.0001">
                    </div>
                </div>
                <div class="row" style="margin-top:10px;">
                    <div class="field">
                        <label>Is in mall?</label>
                        <select id="pred-mall">
                            <option value="false">No</option>
                            <option value="true">Yes</option>
                        </select>
                    </div>
                    <div class="field">
                        <label>Kelurahan ID (optional)</label>
                        <input type="text" id="pred-kel" placeholder="e.g., 5101010001">
                    </div>
                </div>
                <button class="btn" style="margin-top:14px;" onclick="runPredict()">Predict Score</button>
                <pre id="predict-output" class="output" style="margin-top:14px;">Prediction result will appear here.</pre>
            </div>
        </div>

        <!-- Blank Spots -->
        <div id="tab-blank" class="panel">
            <div class="card">
                <h2>Recommended Blank Spot Areas</h2>
                <p>High-score candidate sites with no existing MAA store nearby. Highlighted in green on the LocInsight map.</p>
                <div class="row">
                    <div class="field">
                        <label>Min Score</label>
                        <input type="number" id="bs-min" value="0.6" step="0.05" min="0.3" max="0.9">
                    </div>
                    <div class="field">
                        <label>Limit</label>
                        <input type="number" id="bs-limit" value="20" min="5" max="100">
                    </div>
                    <div class="field">
                        <label>Min distance from MAA store (m)</label>
                        <input type="number" id="bs-radius" value="1000" step="100" min="500" max="3000">
                    </div>
                </div>
                <button class="btn" style="margin-top:14px;" onclick="runBlankSpots()">Find Blank Spots</button>
                <pre id="blank-output" class="output" style="margin-top:14px;">Recommendations will appear here.</pre>
            </div>
        </div>

        <!-- Train -->
        <div id="tab-train" class="panel">
            <div class="card">
                <h2>Train Gradient Boosting Regressor (in-browser)</h2>
                <p>Trains a GBR model on synthetic data using scikit-learn (lazy-loaded via micropip). The trained model is active for this browser session and will be used by Predict and Blank Spots tabs.</p>
                <div class="row">
                    <div class="field">
                        <label>Training samples</label>
                        <input type="number" id="tr-samples" value="500" min="100" max="2000" step="100">
                    </div>
                    <div class="field">
                        <label>N estimators (trees)</label>
                        <input type="number" id="tr-estimators" value="80" min="20" max="200" step="10">
                    </div>
                    <div class="field">
                        <label>Max depth</label>
                        <input type="number" id="tr-depth" value="3" min="2" max="6" step="1">
                    </div>
                </div>
                <button class="btn" style="margin-top:14px;" onclick="runTrain()">Train GBR Model</button>
                <pre id="train-output" class="output" style="margin-top:14px;">Training result will appear here. First click downloads scikit-learn (~5MB, cached after).</pre>
            </div>
            <div class="card">
                <h2>Active Model Info</h2>
                <button class="btn" onclick="runModelInfo()">Show Active Model Metadata</button>
                <pre id="model-info-output" class="output" style="margin-top:14px;">Model metadata will appear here.</pre>
            </div>
        </div>

        <!-- Data Explorer -->
        <div id="tab-data" class="panel">
            <div class="card">
                <h2>Browse Raw Master Data</h2>
                <p>Read-only view of Supabase master tables. RLS allows anon read on these tables.</p>
                <div class="row">
                    <div class="field">
                        <label>Table</label>
                        <select id="de-table">
                            <option value="brands">brands</option>
                            <option value="stores">stores</option>
                            <option value="malls">malls</option>
                            <option value="pois">pois</option>
                            <option value="kelurahan">kelurahan</option>
                            <option value="kabupaten">kabupaten</option>
                            <option value="kecamatan">kecamatan</option>
                            <option value="competitor_stores">competitor_stores</option>
                        </select>
                    </div>
                    <div class="field">
                        <label>Row limit</label>
                        <input type="number" id="de-limit" value="20" min="1" max="500">
                    </div>
                </div>
                <button class="btn" style="margin-top:14px;" onclick="runDataExplorer()">Fetch Data</button>
                <pre id="data-output" class="output" style="margin-top:14px;">Data will appear here.</pre>
            </div>
        </div>

        <!-- About -->
        <div id="tab-about" class="panel">
            <div class="card">
                <h2>LocInsight ML Engine — PyScript Edition</h2>
                <p><strong>What changed from v1?</strong></p>
                <p>v1 used FastAPI + Docker SDK on HF Spaces, which required <code>cpu-basic</code> hardware. HF free tier has <code>cpu-basic</code> quota = 0, so the Space was permanently PAUSED.</p>
                <p>v2 (this version) uses <strong>PyScript</strong> — Python runs in your browser via Pyodide (WebAssembly). The HF Space is served as <strong>static files</strong> (HTML/CSS/JS), which works on the free tier with zero compute quota.</p>
                <p><strong>Trade-offs:</strong></p>
                <ul style="margin-left:20px; font-size:13px; color:#4b5563;">
                    <li>Compute: Client-side (browser) — no server needed</li>
                    <li>API for Vercel: No (browser-only) — Vercel frontend uses its own TS ML engine</li>
                    <li>Cold start: None (static)</li>
                    <li>ML libraries: Pyodide-supported (numpy, scikit-learn via micropip)</li>
                    <li>Cost: Free forever</li>
                    <li>Persistence: localStorage (metadata only) + Supabase read (anon RLS)</li>
                </ul>
                <p style="margin-top:14px;"><strong>Source code:</strong> <a href="https://github.com/bayhaqy/LocInsights_ml" target="_blank">github.com/bayhaqy/LocInsights_ml</a></p>
                <p><strong>Maintained by:</strong> Achmad Bayhaqy — Data Team, MAP Active Adiperkasa (MAA)</p>
            </div>
        </div>
    </div>

    <footer>
        LocInsight ML Engine v2.1.0 &middot; PyScript (Pyodide) &middot; HF Spaces Static SDK<br>
        Data: Supabase (anon RLS) &middot; ML: scikit-learn in-browser &middot; &copy; 2026 MAP Active Adiperkasa
    </footer>

    <!-- PyScript: Python code loaded from app.py via src attribute -->
    <script type="py" src="app.py" config='{"packages":["numpy"]}'></script>

    <!-- JavaScript: UI handlers that call Python functions -->
    <script>
        function showTab(name, btn) {
            document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.getElementById('tab-' + name).classList.add('active');
            btn.classList.add('active');
        }

        function setOutput(id, text) {
            document.getElementById(id).textContent = text;
        }

        function setLoading(id, msg) {
            document.getElementById(id).textContent = '⏳ ' + msg;
        }

        // Health Check
        async function runHealth() {
            setLoading('health-output', 'Running health check...');
            try {
                if (typeof window.health_check !== 'function') {
                    setOutput('health-output', 'Error: Python not ready yet. Wait a few seconds and try again.');
                    return;
                }
                const result = await window.health_check();
                setOutput('health-output', result);
            } catch (e) {
                setOutput('health-output', 'Error: ' + e.message);
            }
        }

        // Predict
        async function runPredict() {
            setLoading('predict-output', 'Fetching data from Supabase + computing features...');
            try {
                if (typeof window.predict_site !== 'function') {
                    setOutput('predict-output', 'Error: Python not ready yet. Wait a few seconds and try again.');
                    return;
                }
                const lat = parseFloat(document.getElementById('pred-lat').value);
                const lng = parseFloat(document.getElementById('pred-lng').value);
                const isMall = document.getElementById('pred-mall').value === 'true';
                const kel = document.getElementById('pred-kel').value;
                const result = await window.predict_site(lat, lng, isMall, kel);
                setOutput('predict-output', result);
            } catch (e) {
                setOutput('predict-output', 'Error: ' + e.message + '\\n\\nStack: ' + (e.stack || ''));
            }
        }

        // Blank Spots
        async function runBlankSpots() {
            setLoading('blank-output', 'Evaluating kelurahan for blank spots (this may take 10-20s)...');
            try {
                if (typeof window.find_blank_spots !== 'function') {
                    setOutput('blank-output', 'Error: Python not ready yet. Wait a few seconds and try again.');
                    return;
                }
                const minScore = parseFloat(document.getElementById('bs-min').value);
                const limit = parseInt(document.getElementById('bs-limit').value);
                const radius = parseInt(document.getElementById('bs-radius').value);
                const result = await window.find_blank_spots(minScore, limit, radius);
                setOutput('blank-output', result);
            } catch (e) {
                setOutput('blank-output', 'Error: ' + e.message);
            }
        }

        // Train
        async function runTrain() {
            setLoading('train-output', 'Loading scikit-learn + training model (first run ~15s)...');
            try {
                if (typeof window.train_model !== 'function') {
                    setOutput('train-output', 'Error: Python not ready yet. Wait a few seconds and try again.');
                    return;
                }
                const samples = parseInt(document.getElementById('tr-samples').value);
                const estimators = parseInt(document.getElementById('tr-estimators').value);
                const depth = parseInt(document.getElementById('tr-depth').value);
                const result = await window.train_model(samples, estimators, depth);
                setOutput('train-output', result);
            } catch (e) {
                setOutput('train-output', 'Error: ' + e.message);
            }
        }

        // Model Info
        async function runModelInfo() {
            setLoading('model-info-output', 'Fetching model metadata...');
            try {
                if (typeof window.model_info !== 'function') {
                    setOutput('model-info-output', 'Error: Python not ready yet. Wait a few seconds and try again.');
                    return;
                }
                const result = window.model_info();
                setOutput('model-info-output', result);
            } catch (e) {
                setOutput('model-info-output', 'Error: ' + e.message);
            }
        }

        // Data Explorer
        async function runDataExplorer() {
            setLoading('data-output', 'Fetching data from Supabase...');
            try {
                if (typeof window.data_explorer !== 'function') {
                    setOutput('data-output', 'Error: Python not ready yet. Wait a few seconds and try again.');
                    return;
                }
                const table = document.getElementById('de-table').value;
                const limit = parseInt(document.getElementById('de-limit').value);
                const result = await window.data_explorer(table, limit);
                setOutput('data-output', result);
            } catch (e) {
                setOutput('data-output', 'Error: ' + e.message);
            }
        }

        // Hide boot loader once PyScript is ready
        window.addEventListener('py:ready', () => {
            const loader = document.getElementById('boot-loader');
            if (loader) {
                loader.classList.add('boot-hidden');
                setTimeout(() => { loader.style.display = 'none'; }, 500);
            }
            setOutput('health-output', 'PyScript ready. Click "Run Health Check" to verify Supabase connectivity.');
        });

        // Fallback: hide loader after 90s even if py:ready doesn't fire
        setTimeout(() => {
            const loader = document.getElementById('boot-loader');
            if (loader && loader.style.display !== 'none') {
                loader.classList.add('boot-hidden');
                setTimeout(() => { loader.style.display = 'none'; }, 500);
            }
        }, 90000);
    </script>
</body>
</html>
"""


def main() -> None:
    if not APP_PY.exists():
        raise SystemExit(f"app.py not found at {APP_PY}")

    # Verify app.py exists and has no </script> in it
    if not APP_PY.exists():
        raise SystemExit(f"app.py not found at {APP_PY}")
    python_source = APP_PY.read_text(encoding="utf-8")
    if "</script>" in python_source.lower():
        raise SystemExit("ERROR: Python source contains '</script>' which would break the HTML")

    # Use src="app.py" instead of inlining. PyScript fetches the .py file
    # at runtime — more reliable than inlining (avoids potential HTML parser
    # edge cases with special characters in Python source).
    html_output = TEMPLATE.replace(PLACEHOLDER, "")

    INDEX_HTML.write_text(html_output, encoding="utf-8")
    size_kb = len(html_output) / 1024
    print(f"Generated {INDEX_HTML} ({size_kb:.1f} KB)")
    print(f"  - app.py source: {len(python_source)} chars")
    print(f"  - Inlined into <script type='py'>")


if __name__ == "__main__":
    main()
