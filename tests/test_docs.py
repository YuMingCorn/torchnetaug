"""Every ```python block in the user-facing docs must run.

Blocks in one file run in order in a shared namespace, like a notebook, so later
examples can use names defined earlier. A block whose first line is ``# doc: skip``
is not run (use it only for snippets that need a real dataset or GPU).
"""

import re
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
DOCS = [
    "README.md",
    "README.en.md",
    "docs/design.md",
    "docs/design.en.md",
    "docs/official_diff.md",
    "docs/official_diff.en.md",
]
BLOCK = re.compile(r"^```python\n(.*?)^```", re.DOTALL | re.MULTILINE)


def blocks(path: Path):
    return [b for b in BLOCK.findall(path.read_text(encoding="utf-8"))]


@pytest.mark.parametrize("doc", DOCS)
def test_doc_examples_run(doc):
    code = blocks(ROOT / doc)
    runnable = [b for b in code if not b.lstrip().startswith("# doc: skip")]
    assert runnable, f"{doc} has no runnable python examples"
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)  # docs show normal float32 usage
    try:
        namespace = {"__name__": "__doc_example__"}
        for i, src in enumerate(runnable):
            try:
                exec(compile(src, f"{doc}[block {i}]", "exec"), namespace)  # noqa: S102
            except Exception as e:
                raise AssertionError(f"{doc} block {i} failed:\n{src}\n{type(e).__name__}: {e}") from e
    finally:
        torch.set_default_dtype(previous)
