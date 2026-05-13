import json
from pathlib import Path

from campus_agent.models import CampusData, Student


DEFAULT_DATA_PATH = Path(__file__).resolve().parents[2] / "campus_data.json"


class CampusDataStore:
    def __init__(self, path: Path | str = DEFAULT_DATA_PATH):
        self.path = Path(path)

    def load(self) -> CampusData:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        return CampusData.model_validate(raw)

    def save(self, data: CampusData) -> None:
        payload = data.model_dump(mode="json")
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def get_student(self, student_id: str) -> Student | None:
        data = self.load()
        for student in data.students:
            if student.student_id == student_id:
                return student
        return None

