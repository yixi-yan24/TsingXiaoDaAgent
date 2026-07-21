import re, json, os
from dataclasses import dataclass, field, asdict
from typing import Optional

MARKDOWN_PATH = os.path.join(os.path.dirname(__file__), "..", "本科辅修培养方案2026版.md")
CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "minors.json")


@dataclass
class MinorProgram:
    name: str
    department: str
    total_credits: str = ""
    prerequisites: str = ""
    major_restrictions: str = ""
    capacity: str = ""
    contact: str = ""
    raw_text: str = ""


def parse_all() -> list[MinorProgram]:
    """Parse the markdown file into structured MinorProgram objects."""
    with open(MARKDOWN_PATH, encoding="utf-8") as f:
        text = f.read()

    lines = text.split("\n")
    minors: list[MinorProgram] = []
    current = None
    buffer = []
    in_program = False

    # Patterns to detect section starts
    prog_pattern = re.compile(r"^##\s+(.+?专业辅修培养方案)$")
    dept_pattern = re.compile(r"^##\s+(.+?学院|.+?系|.+?书院)$")
    section_pattern = re.compile(r"^##\s+")

    for index, line in enumerate(lines):
        stripped = line.strip()

        # Detect minor program start
        prog_match = prog_pattern.search(stripped)
        dept_match = dept_pattern.search(stripped)

        if prog_match:
            # Save previous
            if current and buffer:
                current.raw_text = "\n".join(buffer).strip()
                _extract_metadata(current)
                minors.append(current)
                buffer = []

            name = prog_match.group(1).strip()
            current = MinorProgram(name=name, department=_guess_department(lines, index))
            in_program = True
            buffer = [stripped]
        elif dept_match and not prog_match:
            if current and in_program:
                buffer.append(stripped)
        elif section_pattern.search(stripped) and "辅修" not in stripped and "课程" not in stripped and "培养" not in stripped:
            if current and in_program:
                buffer.append(stripped)
        elif current and in_program:
            buffer.append(stripped)

    # Last one
    if current and buffer:
        current.raw_text = "\n".join(buffer).strip()
        _extract_metadata(current)
        minors.append(current)

    # Also detect the "2026年开放辅修专业一览表" section and get the list
    return minors


def _guess_department(lines: list[str], idx: int) -> str:
    """Look backwards from a program heading to find the department name."""
    for i in range(idx - 1, max(idx - 10, 0), -1):
        m = re.search(r"^##\s+(.+?学院|.+?系|.+?书院)", lines[i].strip())
        if m:
            return m.group(1).strip()
    return ""


def _extract_metadata(prog: MinorProgram):
    """Extract structured metadata from raw_text using regex."""
    text = prog.raw_text

    # Total credits
    m = re.search(r"培养要求\s*(?:总学分|学分)\s*([\d.]+)\s*学分", text)
    if not m:
        m = re.search(r"辅修培养要求\s*([\d.]+)\s*学分", text)
    if not m:
        m = re.search(r"总学分\s*(?:要求|)\s*([\d.]+)\s*学分", text)
    if not m:
        m = re.search(r"([\d.]+)\s*学分", text[:200])
    if m:
        prog.total_credits = m.group(1) + "学分"

    # Prerequisites (先修课程)
    m = re.search(r"先修课程[要求]*[：:](.*?)(?=\n##|\n\d、|\n2[、.)]|\n专业辅修|\n二[、.)]|\n辅修专业)", text, re.DOTALL)
    if m:
        prog.prerequisites = m.group(1).strip()[:500]

    # Major restrictions (主修专业限制)
    # Try the heading followed by paragraph
    m = re.search(r"主修专业[限制面向][^。\n]{0,30}[。\n]\s*([^<>\n][^。]{10,300})", text)
    if not m:
        m = re.search(r"该辅修面向\s*([^<>\n][^。]{10,200})", text)
    if not m:
        m = re.search(r"面向\s*([^<>\n][^。]{20,200})", text)
    if m:
        prog.major_restrictions = re.sub(r'<[^>]+>', '', m.group(1).strip())[:300]

    # Capacity
    m = re.search(r"每年可接纳学生[：:数]*\s*([\d]+)\s*人", text)
    if m:
        prog.capacity = m.group(1) + "人"

    # Contact
    m = re.search(r"(?:咨询)?电话[：:]\s*([\d-]+)", text)
    if m:
        prog.contact = m.group(1).strip()


def load_minors(force_reload: bool = False) -> list[MinorProgram]:
    """Load minors from cache or parse from scratch."""
    if not force_reload and os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, encoding="utf-8") as f:
                data = json.load(f)
            return [MinorProgram(**item) for item in data]
        except Exception:
            pass

    minors = parse_all()
    try:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump([asdict(m) for m in minors], f, ensure_ascii=False, indent=2)
    except PermissionError:
        pass
    return minors


def get_minor_by_name(name: str, minors: list[MinorProgram]) -> Optional[MinorProgram]:
    """Find a minor program by name (partial match)."""
    name = name.strip()
    if not name:
        return None
    for m in minors:
        if name in m.name or m.name in name:
            return m
    for m in minors:
        for kw in name.split():
            if kw in m.name:
                return m
    return None


def search_minors(query: str, minors: list[MinorProgram]) -> list[MinorProgram]:
    """Search minors by keyword in name, department, or raw_text."""
    q = query.strip().lower()
    if not q:
        return []
    scored = []
    for m in minors:
        if q in m.name.lower():
            score = 100
        elif q in m.department.lower():
            score = 70
        elif q in m.prerequisites.lower() or q in m.major_restrictions.lower():
            score = 50
        elif q in m.raw_text.lower():
            score = 30
        else:
            continue
        scored.append((score, m))
    scored.sort(key=lambda item: (-item[0], item[1].name))
    return [minor for _, minor in scored]


def get_all_minor_names(minors: list[MinorProgram]) -> list[str]:
    return [m.name for m in minors]
