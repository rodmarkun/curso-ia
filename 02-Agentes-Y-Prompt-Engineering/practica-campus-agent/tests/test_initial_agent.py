from campus_agent.tools import build_tools


def test_initial_codebase_starts_without_tools():
    assert build_tools() == []
