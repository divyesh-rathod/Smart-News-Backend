

import subprocess
import sys
from pathlib import Path

from app.services import recommendation_cache

def run_subprocess(module_path: str) -> None:
  
    python_bin = sys.executable
    cmd = [python_bin, "-m", module_path]

 
    project_root = Path(__file__).parent.parent.parent

    subprocess.run(cmd, cwd=project_root, check=True)

def start_full_pipeline_subprocesses() -> None:
    try:
        run_subprocess("app.scrapping.scraper")
        run_subprocess("app.preprocessing.preprocess")
        run_subprocess("app.ml_models.generate_embeddings")
    finally:
        # Even a failed run may have added embedded articles.
        recommendation_cache.clear()
