from pathlib import Path


def test_project_files_present():
    assert Path("app_sdcpp.py").exists()
    assert Path("Dockerfile").exists()


def test_readme_mentions_sdcpp_only():
    text = Path("README.md").read_text().lower()
    assert "sd.cpp only" in text or "stable-diffusion.cpp" in text
    assert "comfyui" not in text
