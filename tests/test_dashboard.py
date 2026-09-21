from streamlit.testing.v1 import AppTest
from pathlib import Path


def test_dashboard_renders_without_agent_call():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=20).run()
    assert not app.exception
    assert app.title[0].value == "CFPB Complaint Intelligence Agent"
    assert len(app.metric) == 4
