from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, RedirectResponse
import uvicorn
import json
import threading
from pathlib import Path
from typing import Dict, Any

from ..trainer.engine import Trainer

import psutil
import time

app = FastAPI()

# Global State (Injected by CLI)
SCENARIO = None
CONFIG_MGR = None
RUN_DIR = Path(".gradia_logs").resolve()
DEFAULT_CONFIG = {}
TRAINER = None
TRAINING_THREAD = None
SYSTEM_THREAD = None

# Mounts
BASE_DIR = Path(__file__).resolve().parent

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
# Mount assets if they exist outside static, or ensure user put them in static. Assuming viz/assets
assets_path = BASE_DIR / "assets"
if assets_path.exists():
    app.mount("/assets", StaticFiles(directory=assets_path), name="assets")

templates = Jinja2Templates(directory=BASE_DIR / "templates")

from ..trainer.callbacks import log_lock

# ... imports ...
import os

# System Monitor
def system_monitor_loop():
    log_path = RUN_DIR / "events.jsonl"
    while True:
        cpu = psutil.cpu_percent(interval=1)
        mem = psutil.virtual_memory().percent
        t = time.time()
        
        event = {
            "timestamp": t,
            "type": "system_metrics", 
            "data": {"cpu": cpu, "ram": mem, "epoch": t} 
        }
        
        if RUN_DIR.exists():
            with log_lock:
                with open(log_path, "a") as f:
                    f.write(json.dumps(event) + "\n")
                    f.flush()
                    os.fsync(f.fileno())

# Start System Monitor on import/startup (or when server starts)
@app.on_event("startup")
async def startup_event():
    global SYSTEM_THREAD
    SYSTEM_THREAD = threading.Thread(target=system_monitor_loop, daemon=True)
    SYSTEM_THREAD.start()


@app.get("/")
async def read_root(request: Request):
    if TRAINER is None:
        return RedirectResponse("/configure")
    return templates.TemplateResponse("index.html", {"request": request, "scenario": SCENARIO})

@app.get("/configure")
async def configure_page(request: Request):
    if SCENARIO is None:
        return "System not initialized correctly from CLI."
        
    return templates.TemplateResponse("configure.html", {
        "request": request, 
        "scenario": SCENARIO,
        "features": SCENARIO.features,
        "default_config": DEFAULT_CONFIG
    })

@app.post("/api/start")
async def start_training(config_data: Dict[str, Any]):
    global TRAINER, TRAINING_THREAD, SCENARIO, RUN_DIR
    
    # Merge received config with defaults
    # Expect config_data = {model: {type:..., params:...}, training: {epochs:...}}
    
    # We construct the full config object
    full_config = DEFAULT_CONFIG.copy()
    
    # Helper to merge deep dicts if needed, or just overwrite keys
    full_config['model'] = config_data.get('model', full_config['model'])
    full_config['training'].update(config_data.get('training', {}))
    
    # New fields
    full_config['project_name'] = config_data.get('project_name', 'experiment')
    full_config['save_model'] = config_data.get('save_model', False)
    
    # Save config
    CONFIG_MGR.save(full_config)
    
    # Initialize Trainer
    TRAINER = Trainer(SCENARIO, full_config, str(RUN_DIR))
    
    # Start Thread
    def train_wrapper():
        import time
        time.sleep(1) # Breathe
        try:
            TRAINER.run()
        except Exception as e:
            print(f"Training Error: {e}")

    TRAINING_THREAD = threading.Thread(target=train_wrapper, daemon=True)
    TRAINING_THREAD.start()
    
    return {"status": "started"}

@app.get("/api/events")
async def get_events():
    event_path = RUN_DIR / "events.jsonl"
    events = []
    
    if event_path.exists():
        # No lock needed for reading usually if we tolerate partial lines (which json.loads handles with try/except)
        # But to be safe vs partial writes, we could lock, but that might block writers.
        # Standard polling read is usually fine without lock if we just read lines.
        with open(event_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    
    return JSONResponse(content=events)

@app.get("/api/report/json")
async def download_report_json():
    event_path = RUN_DIR / "events.jsonl"
    if not event_path.exists():
        return JSONResponse({"error": "No logs found"}, status_code=404)
    
    events = []
    with open(event_path, "r") as f:
        for line in f:
            if line.strip():
                try: events.append(json.loads(line))
                except: pass
                
    return JSONResponse(content={"project": SCENARIO.target_column if SCENARIO else "gradia", "events": events})

@app.get("/api/report/pdf")
async def download_report_pdf():
    # Return a HTML page optimized for print-to-pdf for simplicity without reportlab dep
    event_path = RUN_DIR / "events.jsonl"
    events = []
    if event_path.exists():
        with open(event_path, "r") as f:
            for line in f:
                try: events.append(json.loads(line))
                except: pass
                
    html = f"""
    <html>
    <head>
        <title>Training Report</title>
        <style>
            body {{ font-family: sans-serif; padding: 40px; }}
            h1 {{ border-bottom: 2px solid #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .metric {{ color: #0066cc; font-weight: bold; }}
        </style>
    </head>
    <body onload="window.print()">
        <h1>Gradia Training Report</h1>
        <p>Target: {SCENARIO.target_column if SCENARIO else 'N/A'}</p>
        <p>Total Epochs: {len([e for e in events if e['type'] == 'epoch_end'])}</p>
        
        <h2>Training History</h2>
        <table>
            <thead><tr><th>Epoch</th><th>Train Acc/MSE</th><th>Test Acc/MSE</th><th>CPU %</th><th>RAM %</th></tr></thead>
            <tbody>
    """
    
    # Process events to correlate metrics
    epochs = [e for e in events if e['type'] == 'epoch_end']
    for e in epochs:
        d = e['data']
        # Find close system metric
        html += f"<tr><td>{d['epoch']}</td><td>{d.get('train_acc', d.get('train_mse', 'N/A'))}</td><td>{d.get('test_acc', d.get('test_mse', 'N/A'))}</td><td>-</td><td>-</td></tr>"
        
    html += """
            </tbody>
        </table>
    </body>
    </html>
    """
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html)

@app.post("/api/evaluate")
async def evaluate_model():
    if TRAINER is None:
        return JSONResponse({"error": "No model trained"}, status_code=400)
    
    try:
        results = TRAINER.evaluate_full()
        return JSONResponse(content=results)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

def start_server(run_dir: str, port: int = 8000):
    global RUN_DIR
    RUN_DIR = Path(run_dir).resolve()
    print(f"DEBUG: Server using RUN_DIR: {RUN_DIR}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="error")
