from pathlib import Path


def test_project_scaffold_exists() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts" / "download_cfpb_sample.py").exists()
    assert (root / "scripts" / "profile_cfpb_data.py").exists()
    assert (root / ".env.example").exists()
