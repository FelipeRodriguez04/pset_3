from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=1800,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
        allow_errors=False,
    )
    client.execute()
    nbformat.write(notebook, path)
    print(f"OK Restart + Run All: {path.name}")
