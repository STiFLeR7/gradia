import typer
import threading
import time
import os
import webbrowser
from pathlib import Path
from rich.console import Console
from ..core.inspector import Inspector
from ..core.scenario import ScenarioInferrer
from ..core.config import ConfigManager
from ..trainer.engine import Trainer
from ..viz import server

app = typer.Typer()
console = Console()

@app.callback()
def callback():
    """
    gradia: Local-first ML training visualization.
    """

@app.command()
def run(
    ctx: typer.Context, 
    path: str = typer.Argument(".", help="Path to data directory"),
    target: str = typer.Option(None, help="Manually specify target column"),
    port: int = typer.Option(8000, help="Port for visualization server")
):
    """
    Starts the gradia training and visualization session.
    """
    console.rule("[bold blue]gradia v1.0.0[/bold blue]")
    
    # 1. Inspect
    path = Path(path).resolve()
    inspector = Inspector(path)
    datasets = inspector.find_datasets()
    
    if not datasets:
        console.print(f"[red]No .csv or .parquet files found in {path}[/red]")
        raise typer.Exit(code=1)
    
    # Select first dataset for MVP
    dataset = datasets[0]
    console.print(f"[green]Found dataset:[/green] {dataset.name}")
    
    # 2. Config & Scenario Reuse
    run_dir = path / ".gradia_logs"
    config_mgr = ConfigManager(run_dir)
    config = config_mgr.load_or_create()
    
    # We infer scenario here to pass to server, but user confirms/configures in UI
    with console.status("Inferring scenario..."):
        inferrer = ScenarioInferrer()
        scenario = inferrer.infer(str(dataset), target_override=target)
    
    console.print(f"Target: [bold]{scenario.target_column}[/bold] | Task: [bold]{scenario.task_type}[/bold]")
    # Session Isolation: Create unique run directory
    session_id = int(time.time())
    run_dir = Path(".gradia_logs") / f"run_{session_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config_mgr = ConfigManager(run_dir)
    config = config_mgr.load_or_create()
    
    # Apply Smart Recommendation
    config['model']['type'] = scenario.recommended_model
    console.print(f"[cyan]Smart Suggestion:[/cyan] Using [bold]{scenario.recommended_model}[/bold] for this dataset.")
    
    console.print(f"[bold green]Configuration moved to Web UI[/bold green]")
    console.print(f"Visualization running at http://localhost:{port}")
    console.print(f"Logs: {run_dir.resolve()}")
    
    # 3. Launch Server
    # We inject state into the server module before starting it
    server.SCENARIO = scenario
    server.CONFIG_MGR = config_mgr
    server.RUN_DIR = run_dir
    server.DEFAULT_CONFIG = config
    
    
    # Launch browser
    threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{port}/configure")).start()
    
    # Start server (blocking main thread is fine now as we don't have a separate training thread YET)
    # The training thread will be spawned by the server upon API request.
    server.start_server(str(run_dir), port)

if __name__ == "__main__":
    app()
