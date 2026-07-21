import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional


COURSES_PATH = os.path.join(os.path.dirname(__file__), "..", "curated_courses.json")


def _normalize(value: str) -> str:
    return re.sub(r"[\s（）()\[\]【】,，.。:：;；/\\_-]+", "", value).lower()


@dataclass
class CourseRecord:
    id: str
    name: str
    department: str = ""
    credits: Optional[float] = None
    total_hours: Optional[int] = None
    prerequisites: str = ""
    description: str = ""
    objectives: str = ""
    expected_outcomes: str = ""
    assessment_method: str = ""
    grade_breakdown: str = ""
    textbooks: str = ""
    instructor: str = ""
    minor_programs: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CourseRecord":
        return cls(
            id=str(data.get("id", "")).strip(),
            name=str(data.get("name", "")).strip(),
            department=str(data.get("department", "")).strip(),
            credits=data.get("credits"),
            total_hours=data.get("total_hours"),
            prerequisites=str(data.get("prerequisites", "")).strip(),
            description=str(data.get("description", "")).strip(),
            objectives=str(data.get("objectives", "")).strip(),
            expected_outcomes=str(data.get("expected_outcomes", "")).strip(),
            assessment_method=str(data.get("assessment_method", "")).strip(),
            grade_breakdown=str(data.get("grade_breakdown", "")).strip(),
            textbooks=str(data.get("textbooks", "")).strip(),
            instructor=str(data.get("instructor", "")).strip(),
            minor_programs=list(data.get("minor_programs") or []),
        )


class CourseCatalog:
    """Searchable local catalog built from the curated minor-course dataset."""

    def __init__(self, courses: list[CourseRecord]):
        self.courses = [course for course in courses if course.id and course.name]
        self._by_id = {course.id: course for course in self.courses}

    @classmethod
    def load(cls, path: str = COURSES_PATH) -> "CourseCatalog":
        try:
            with open(path, encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"无法加载课程目录: {path}") from exc
        if not isinstance(data, list):
            raise RuntimeError("课程目录格式错误：顶层必须是列表")
        return cls([CourseRecord.from_dict(item) for item in data if isinstance(item, dict)])

    def find(self, identifier: str) -> Optional[CourseRecord]:
        query = _normalize(identifier)
        if not query:
            return None
        if identifier.strip() in self._by_id:
            return self._by_id[identifier.strip()]
        exact = [course for course in self.courses if _normalize(course.name) == query]
        if len(exact) == 1:
            return exact[0]
        partial = [course for course in self.courses if query in _normalize(course.name)]
        return partial[0] if len(partial) == 1 else None

    def search(self, query: str, limit: int = 5) -> list[CourseRecord]:
        normalized_query = _normalize(query)
        if not normalized_query:
            return []

        scored: list[tuple[int, CourseRecord]] = []
        for course in self.courses:
            name = _normalize(course.name)
            program_names = " ".join(
                str(program.get("program", "")) for program in course.minor_programs
            )
            metadata = _normalize(f"{course.department} {program_names}")
            content = _normalize(
                f"{course.description[:2000]} {course.objectives[:1000]} "
                f"{course.expected_outcomes[:1000]} {course.prerequisites}"
            )

            score = 0
            if course.id == query.strip():
                score = 100
            elif name == normalized_query:
                score = 95
            elif normalized_query in name:
                score = 80
            elif normalized_query in metadata:
                score = 55
            elif normalized_query in content:
                score = 35
            if score:
                scored.append((score, course))

        scored.sort(key=lambda item: (-item[0], item[1].name, item[1].id))
        return [course for _, course in scored[:limit]]

    def for_minor(self, minor_name: str, limit: int = 40) -> list[CourseRecord]:
        query = _normalize(minor_name.replace("专业辅修培养方案", ""))
        if not query:
            return []
        matches = []
        for course in self.courses:
            if any(query in _normalize(str(program.get("program", "")))
                   for program in course.minor_programs):
                matches.append(course)
        return matches[:limit]
