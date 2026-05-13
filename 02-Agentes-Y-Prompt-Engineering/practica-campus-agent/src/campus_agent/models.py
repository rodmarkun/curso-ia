from pydantic import BaseModel, Field


class Subject(BaseModel):
    code: str
    name: str
    grade: float | None = None
    status: str


class Student(BaseModel):
    student_id: str
    name: str
    email: str
    degree: str
    subjects: list[Subject] = Field(default_factory=list)


class CampusData(BaseModel):
    students: list[Student] = Field(default_factory=list)
