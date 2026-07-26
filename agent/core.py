import json, re
from collections.abc import Generator

import httpx

from .llm_client import chat_completion, chat_completion_stream
from .memory import ShortTermMemory, LongTermMemory
from .tools import Tools
from .planner import CoursePlanner
from .prompts import SYSTEM_PROMPT
from .data_loader import load_minors


class MinorAdvisorAgent:
    """The main agent orchestrator for Tsinghua Minor Program advising."""

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/v1"):
        self.api_key = api_key
        self.base_url = base_url

        # Load long-term memory (minor program database)
        minors = load_minors()
        self.ltm = LongTermMemory(minors)

        # Initialize tools
        self.tools = Tools(self.ltm, api_key=api_key)

        # Initialize planner
        self.planner = CoursePlanner(api_key, base_url)

        # Initialize short-term memory (per-session, will be copied for each session)
        self._default_stm = ShortTermMemory()

    def create_session(self) -> "AgentSession":
        """Create a new conversation session."""
        return AgentSession(
            api_key=self.api_key,
            base_url=self.base_url,
            ltm=self.ltm,
            tools=self.tools,
            planner=self.planner
        )


class AgentSession:
    """A single conversation session with its own short-term memory."""

    # ── tool dispatch table (built once per class) ──────────────────────
    _TOOL_PARAM_MAP: dict[str, list[str]] = {
        "list_minors": [],
        "search_minors": ["keyword"],
        "get_minor_detail": ["name"],
        "search_courses": ["keyword"],
        "get_course_detail": ["identifier"],
        "list_minor_courses": ["minor_name"],
        "check_eligibility": ["major", "minor_name"],
        "semantic_search": ["query"],
        "multi_agent_search": ["major", "interests", "grade"],
    }

    def __init__(
        self,
        api_key: str,
        base_url: str,
        ltm: LongTermMemory,
        tools: Tools,
        planner: CoursePlanner
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.ltm = ltm
        self.tools = tools
        self.planner = planner
        self.stm = ShortTermMemory()
        self.stm.add("system", SYSTEM_PROMPT)
        # Lazily-built tool block (cached so recursive _call_llm reuses it).
        self._tool_block: str | None = None

    # ── public API ──────────────────────────────────────────────────────

    def process_message(self, user_message: str, temperature: float = 0.3) -> str:
        """Process a user message and return the agent response."""
        self.stm.add("user", user_message)

        # Get response from LLM
        response = self._call_llm(temperature=temperature)
        self.stm.add("assistant", response)
        return response

    def process_message_stream(
        self, user_message: str, temperature: float = 0.3
    ) -> Generator[str, None, None]:
        """Like process_message but yields tokens for the **final** response.

        Internal tool-call loops are still non-streaming (the full response is
        needed to parse ACTION / PARAMS), but the last LLM turn uses true
        SSE streaming so the user sees tokens progressively.
        """
        self.stm.add("user", user_message)
        full_response: list[str] = []
        for token in self._call_llm_stream(temperature=temperature):
            full_response.append(token)
            yield token
        self.stm.add("assistant", "".join(full_response))

    def process_with_planning(self, major: str, grade: str, minor_name: str) -> str:
        """Generate a course plan for a specific minor."""
        plan = self.planner.generate_plan(major, grade, minor_name, self.ltm)
        self.stm.add("user", f"请为{major}专业{grade}学生制定{minor_name}辅修修读计划")
        self.stm.add("assistant", plan)
        return plan

    def get_history(self) -> list[dict]:
        return self.stm.to_llm_format()

    def clear(self):
        self.stm.clear()
        self.stm.add("system", SYSTEM_PROMPT)

    # ── internal ────────────────────────────────────────────────────────

    def _get_tool_block(self) -> str:
        """Build (and cache) the tool-use instructions block."""
        if self._tool_block is not None:
            return self._tool_block
        tool_descriptions = self.tools.get_tool_descriptions()
        block = "\n\n你可以在回答前使用以下工具获取信息。如果需要使用工具，输出格式为：\n"
        block += "THOUGHT: <你的思考过程>\n"
        block += "ACTION: <工具名称>\n"
        block += "PARAMS: {\"参数名\": \"参数值\"}\n\n"
        block += "工具列表：\n"
        for t in tool_descriptions:
            block += f"- {t['name']}: {t['description']}\n"
            if t['parameters']:
                for pname, pinfo in t['parameters'].items():
                    block += f"  参数 {pname}: {pinfo.get('description', '')}\n"
        self._tool_block = block
        return block

    def _call_llm(self, temperature: float = 0.3, tool_calls: int = 0) -> str:
        """Call DeepSeek API and handle tool use via prompt-based function calling."""
        messages = self.stm.to_llm_format()
        tool_block = self._get_tool_block()

        # Attach tool instructions to the system message (STM stays untouched).
        augmented_messages = []
        for msg in messages:
            if msg["role"] == "system":
                augmented_messages.append({
                    "role": "system",
                    "content": msg["content"] + "\n" + tool_block
                })
            else:
                augmented_messages.append(msg)

        content = chat_completion(
            self.api_key, self.base_url, augmented_messages,
            temperature=temperature, max_tokens=4096, timeout=90, retries=1,
        )

        # Tool-use loop (max 3 hops).
        tool_result = self._parse_tool_call(content)
        if tool_result:
            if tool_calls >= 3:
                return "抱歉，查询所需的工具调用次数过多。请缩小问题范围后重试。"
            tool_name, params = tool_result
            result = self._execute_tool(tool_name, params)
            self.stm.add("tool", result, tool_name=tool_name)
            return self._call_llm(temperature, tool_calls + 1)

        return self._clean_response(content)

    def _call_llm_stream(
        self, temperature: float = 0.3, tool_calls: int = 0
    ) -> Generator[str, None, None]:
        """Streaming variant of _call_llm.

        Uses true SSE streaming from DeepSeek.  Tokens are buffered internally
        so tool calls can still be parsed from the complete response.  Only the
        **final** turn (no tool call) yields tokens to the caller — earlier
        turns are consumed silently.
        """
        messages = self.stm.to_llm_format()
        tool_block = self._get_tool_block()

        augmented_messages = []
        for msg in messages:
            if msg["role"] == "system":
                augmented_messages.append({
                    "role": "system",
                    "content": msg["content"] + "\n" + tool_block
                })
            else:
                augmented_messages.append(msg)

        # Stream from DeepSeek — buffer tokens so we can inspect for tool calls.
        buffered: list[str] = []
        for token in chat_completion_stream(
            self.api_key, self.base_url, augmented_messages,
            temperature=temperature, max_tokens=4096, timeout=90,
        ):
            buffered.append(token)

        content = "".join(buffered)

        # Check if the LLM wants to use a tool
        tool_result = self._parse_tool_call(content)
        if tool_result:
            if tool_calls >= 3:
                yield "抱歉，查询所需的工具调用次数过多。请缩小问题范围后重试。"
                return
            tool_name, params = tool_result
            result = self._execute_tool(tool_name, params)
            self.stm.add("tool", result, tool_name=tool_name)
            yield from self._call_llm_stream(temperature, tool_calls + 1)
            return

        # Final turn — yield cleaned content character by character.
        content = self._clean_response(content)
        for char in content:
            yield char

    def _clean_response(self, content: str) -> str:
        """Remove internal THOUGHT: prefix if present and not followed by a tool call."""
        lines = content.split("\n")
        # Pre-compute: are there ACTION lines?  If yes, keep everything for parsing.
        has_action = any(l.strip().startswith("ACTION:") for l in lines)
        if has_action:
            return content
        # No tool call — drop every THOUGHT: line.
        cleaned = [l for l in lines if not l.strip().startswith("THOUGHT:")]
        result = "\n".join(cleaned).strip()
        return result if result else content

    def _parse_tool_call(self, content: str):
        """Parse THOUGHT / ACTION / PARAMS from LLM output."""
        action_match = re.search(r"ACTION:\s*(\w+)", content)
        params_match = re.search(r"PARAMS:\s*(\{.*?\})", content, re.DOTALL)
        if action_match:
            tool_name = action_match.group(1)
            params = {}
            if params_match:
                try:
                    params = json.loads(params_match.group(1))
                except json.JSONDecodeError:
                    pass
            return tool_name, params
        return None

    def _execute_tool(self, tool_name: str, params: dict) -> str:
        """Execute a tool and return its result."""
        tool_map = {
            "list_minors": lambda: self.tools.list_minors(),
            "search_minors": lambda: self.tools.search_minors(params.get("keyword", "")),
            "get_minor_detail": lambda: self.tools.get_minor_detail(params.get("name", "")),
            "search_courses": lambda: self.tools.search_courses(params.get("keyword", "")),
            "get_course_detail": lambda: self.tools.get_course_detail(params.get("identifier", "")),
            "list_minor_courses": lambda: self.tools.list_minor_courses(params.get("minor_name", "")),
            "check_eligibility": lambda: self.tools.check_eligibility(
                params.get("major", ""), params.get("minor_name", "")
            ),
            "semantic_search": lambda: self.tools.semantic_search(params.get("query", "")),
            "multi_agent_search": lambda: self.tools.multi_agent_search(
                params.get("major", ""), params.get("interests", ""), params.get("grade", "")
            ),
        }
        func = tool_map.get(tool_name)
        if func:
            return func()
        return f"未知工具: {tool_name}"
