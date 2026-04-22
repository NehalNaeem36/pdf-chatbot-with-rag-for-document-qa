from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def test_modules_import() -> None:
    sys.path.insert(0, str(PROJECT_DIR))

    import app  # noqa: F401
    import chunker  # noqa: F401
    import embedder  # noqa: F401
    import pdf_reader  # noqa: F401
    import qa_engine  # noqa: F401
    import reranker  # noqa: F401
    import retriever  # noqa: F401
    import scope_checker  # noqa: F401


def test_main_runs_without_pdf() -> None:
    result = subprocess.run(
        [sys.executable, str(PROJECT_DIR / "main.py")],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "PDF QA skeleton is ready" in result.stdout
