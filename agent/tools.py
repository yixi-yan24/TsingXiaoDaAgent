from typing import Optional
from .course_catalog import CourseCatalog, CourseRecord
from .memory import LongTermMemory


class Tools:
    """Tools available to the agent."""

    def __init__(self, long_term_memory: LongTermMemory, api_key: str = "", course_catalog: Optional[CourseCatalog] = None):
        self.ltm = long_term_memory
        self._api_key = api_key

    def list_minors(self, group_by_department: bool = True) -> str:
        """列出所有可用的辅修专业（按院系分组）。"""
        names = self.ltm.list_all()
        if not group_by_department:
            return "清华大学2026年开放辅修专业列表：\n" + "\n".join(f"  {i+1}. {n}" for i, n in enumerate(names))

        # Group by department for compact, scannable output.
        from collections import defaultdict
        grouped: dict[str, list[str]] = defaultdict(list)
        for m in self.ltm.minors:
            dept = m.department or "其他院系"
            grouped[dept].append(m.name)

        lines = ["清华大学2026年开放辅修专业列表（按院系分组）："]
        for dept, minors in sorted(grouped.items()):
            lines.append(f"\n【{dept}】")
            for name in minors:
                lines.append(f"  - {name}")
        lines.append(f"\n共 {len(names)} 个辅修专业。")
        return "\n".join(lines)

    def search_minors(self, keyword: str) -> str:
        """搜索辅修专业。"""
        if not keyword.strip():
            return "请提供专业名称、院系或方向关键词后再搜索。"
        results = self.ltm.search(keyword)
        if not results:
            return f"未找到与 '{keyword}' 相关的辅修专业。"
        lines = [f"找到 {len(results)} 个相关辅修专业："]
        for m in results:
            lines.append(f"\n【{m.name}】({m.department})")
            lines.append(f"  学分要求：{m.total_credits or '见培养方案'}")
            lines.append(f"  主修限制：{m.major_restrictions or '见培养方案'}")
            if m.capacity:
                lines.append(f"  接纳人数：{m.capacity}")
            if m.contact:
                lines.append(f"  咨询电话：{m.contact}")
        return "\n".join(lines)

    def search_courses(self, keyword: str) -> str:
        """搜索已收录的辅修课程，可按课程号、名称、院系或课程内容关键词查询。"""
        if not keyword.strip():
            return "请提供课程号、课程名称或主题关键词后再搜索。"
        courses = self.course_catalog.search(keyword)
        if not courses:
            return f"未在已收录的辅修课程资料中找到与“{keyword}”相关的课程。"
        lines = [f"找到 {len(courses)} 门相关课程："]
        for course in courses:
            programs = "、".join(
                str(program.get("program", "")).replace("专业辅修培养方案", "")
                for program in course.minor_programs[:3]
            )
            lines.append(
                f"\n【{course.name}】课程号：{course.id}｜{course.department}｜"
                f"{course.credits if course.credits is not None else '未知'} 学分"
            )
            if programs:
                lines.append(f"  关联辅修：{programs}")
            if course.prerequisites:
                lines.append(f"  先修要求：{course.prerequisites[:200]}")
            if course.description:
                lines.append(f"  内容摘要：{course.description[:240]}")
        return "\n".join(lines)

    def get_course_detail(self, identifier: str) -> str:
        """获取一门已收录辅修课程的课程内容、先修、考核与教材信息。"""
        course = self.course_catalog.find(identifier)
        if not course:
            matches = self.course_catalog.search(identifier, limit=2) if identifier.strip() else []
            if matches:
                return "课程名称不够明确，请从以下候选中指定一门：" + "、".join(
                    f"{item.name}（{item.id}）" for item in matches
                )
            return f"未找到课程：{identifier}"
        return self._format_course_detail(course)

    def list_minor_courses(self, minor_name: str) -> str:
        """列出某辅修培养方案中已收录详细资料的课程。"""
        program = self.ltm.find_minor(minor_name)
        if not program:
            return f"未找到辅修专业: {minor_name}"
        courses = self.course_catalog.for_minor(program.name)
        if not courses:
            return f"{program.name} 暂无可用的详细课程资料。"
        lines = [f"{program.name} 已收录详细资料的课程（共 {len(courses)} 门）："]
        for course in courses:
            lines.append(
                f"- {course.name}（{course.id}，{course.credits if course.credits is not None else '未知'} 学分，{course.department}）"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_course_detail(course: CourseRecord) -> str:
        programs = "、".join(
            str(program.get("program", "")).replace("专业辅修培养方案", "")
            for program in course.minor_programs
        ) or "未标注"
        parts = [
            f"【{course.name}】",
            f"课程号：{course.id}",
            f"开课单位：{course.department or '未标注'}",
            f"学分：{course.credits if course.credits is not None else '未标注'}",
            f"总学时：{course.total_hours if course.total_hours is not None else '未标注'}",
            f"关联辅修：{programs}",
        ]
        fields = [
            ("先修要求", course.prerequisites),
            ("课程内容", course.description),
            ("教学目标", course.objectives),
            ("预期学习成效", course.expected_outcomes),
            ("考核方式", course.assessment_method),
            ("成绩构成", course.grade_breakdown),
            ("教材及参考书", course.textbooks),
            ("课程负责人", course.instructor),
        ]
        for label, value in fields:
            if value:
                parts.append(f"\n{label}：{value[:1200]}")
        return "\n".join(parts)

    def get_minor_detail(self, name: str) -> str:
        """获取某个辅修专业的详细信息。"""
        prog = self.ltm.find_minor(name)
        if not prog:
            return f"未找到辅修专业: {name}"
        parts = [
            f"【{prog.name}】",
            f"开设院系：{prog.department}",
            f"学分要求：{prog.total_credits or '见培养方案'}",
        ]
        if prog.prerequisites:
            parts.append(f"先修课程：{prog.prerequisites}")
        if prog.major_restrictions:
            parts.append(f"主修专业限制：{prog.major_restrictions}")
        if prog.capacity:
            parts.append(f"每年可接纳：{prog.capacity}")
        if prog.contact:
            parts.append(f"咨询电话：{prog.contact}")
        parts.append(f"\n--- 培养方案详情 ---\n{prog.raw_text[:3000]}")
        return "\n".join(parts)

    def check_eligibility(self, major: str, minor_name: str) -> str:
        """检查学生主修专业是否符合辅修申请条件。"""
        prog = self.ltm.find_minor(minor_name)
        if not prog:
            return f"未找到辅修专业: {minor_name}"
        restrictions = prog.major_restrictions or "未明确列出，建议直接咨询院系确认。"
        return (
            f"【资格检查】主修：{major} → 辅修：{prog.name}\n"
            f"限制说明：{restrictions}\n"
            f"注意：辅修学位要求与主修专业归属不同专业类，请确认二者的专业类关系。"
        )

    def multi_agent_search(self, major: str, interests: str, grade: str = "") -> str:
        """【Multi-Agent 协同搜索】使用多个专业子 Agent 协同分析学生需求，推荐最适配的辅修专业。"""
        try:
            from .multi_agent import MultiAgentSystem
            # Cache the system so sub-agents are reused across calls.
            if self._multi_agent_system is None:
                self._multi_agent_system = MultiAgentSystem(api_key=self._api_key)
            mas = self._multi_agent_system
            profile = {"major": major, "grade": grade, "interests": interests}
            return mas.search_recommendations(profile, self.ltm)
        except Exception as e:
            return f"Multi-Agent 搜索出错: {e}"

    def get_tool_descriptions(self) -> list[dict]:
        """Return tool descriptions in a format usable by LLM."""
        return [
            {
                "name": "list_minors",
                "description": "列出所有清华大学2026年开放辅修专业",
                "parameters": {}
            },
            {
                "name": "search_minors",
                "description": "根据关键词搜索辅修专业",
                "parameters": {
                    "keyword": {"type": "string", "description": "搜索关键词，如专业名称、院系"}
                }
            },
            {
                "name": "get_minor_detail",
                "description": "获取某个辅修专业的详细培养方案",
                "parameters": {
                    "name": {"type": "string", "description": "辅修专业名称"}
                }
            },
            {
                "name": "search_courses",
                "description": "按课程号、课程名称、院系或课程主题搜索已收录的辅修课程",
                "parameters": {
                    "keyword": {"type": "string", "description": "例如：机器学习、建筑设计、30000833"}
                }
            },
            {
                "name": "get_course_detail",
                "description": "获取一门课程的内容简介、先修要求、考核方式和教材等详细资料",
                "parameters": {
                    "identifier": {"type": "string", "description": "精确课程号或明确课程名称"}
                }
            },
            {
                "name": "list_minor_courses",
                "description": "列出某辅修专业中已收录详细课程资料的课程",
                "parameters": {
                    "minor_name": {"type": "string", "description": "辅修专业名称"}
                }
            },
            {
                "name": "check_eligibility",
                "description": "检查学生主修专业是否符合辅修申请条件",
                "parameters": {
                    "major": {"type": "string", "description": "学生的主修专业"},
                    "minor_name": {"type": "string", "description": "辅修专业名称"}
                }
            },
            {
                "name": "semantic_search",
                "description": "【词嵌入语义搜索】用 AI 理解查询意图进行语义搜索（如搜索'计算机'也能找到'软件工程'、'人工智能'等）",
                "parameters": {
                    "query": {"type": "string", "description": "搜索查询"}
                }
            },
            {
                "name": "multi_agent_search",
                "description": "【Multi-Agent】使用多个AI专家协同分析，推荐最适配的辅修专业",
                "parameters": {
                    "major": {"type": "string", "description": "学生的主修专业"},
                    "interests": {"type": "string", "description": "学生的兴趣方向"},
                    "grade": {"type": "string", "description": "年级"}
                }
            }
        ]
