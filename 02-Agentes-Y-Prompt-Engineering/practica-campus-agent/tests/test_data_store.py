from campus_agent.data_store import CampusDataStore


def test_loads_seed_data():
    store = CampusDataStore()
    data = store.load()

    assert len(data.students) >= 2
    assert data.students[0].student_id.startswith("alu-")

