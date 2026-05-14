import json
from pathlib import Path


def test_project_files_present():
    assert Path("app.py").exists()
    assert Path("comfyui_flux2_gguf_api_workflow.json").exists()


def test_workflow_json_valid():
    wf = json.loads(Path("comfyui_flux2_gguf_api_workflow.json").read_text())
    assert isinstance(wf, dict)
    assert len(wf) > 0
