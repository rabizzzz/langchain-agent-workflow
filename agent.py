"""
LangChain-style Agentic Workflow
Multi-step reasoning agent with tool use, memory, and evaluation.
"""

import logging
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AGENT_PROMPT = """You are a helpful AI assistant with access to tools.

Available tools: {tools}

To use a tool, respond with:
TOOL: <tool_name>
INPUT: <tool_input>

When you have a final answer, respond with:
FINAL: <your_answer>

Context: {context}
User request: {query}

Begin reasoning:"""


@dataclass
class AgentStep:
    step_number: int
    thought: str
    action: str = ""
    tool_name: str = ""
    tool_output: str = ""
    is_final: bool = False


@dataclass
class AgentResult:
    query: str
    final_answer: str
    steps_taken: list = field(default_factory=list)
    total_steps: int = 0
    success: bool = True
    error: str = ""


class ToolRegistry:
    """Registry of tools available to the agent."""

    def __init__(self):
        self._tools = {
            "calculator": self._calculator,
            "summarizer": self._summarizer,
            "web_search": self._web_search,
            "fact_checker": self._fact_checker,
        }

    def invoke(self, tool_name, tool_input):
        if tool_name not in self._tools:
            return f"Unknown tool: {tool_name}"
        try:
            return self._tools[tool_name](tool_input)
        except Exception as e:
            return f"Tool error: {str(e)}"

    def list_tools(self):
        return list(self._tools.keys())

    def _calculator(self, expression):
        allowed = set("0123456789+-*/()., ")
        if not all(c in allowed for c in expression):
            raise ValueError("Invalid expression")
        return str(eval(expression, {"__builtins__": {}}))

    def _summarizer(self, text):
        sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 20]
        n = max(1, len(sentences) // 3)
        return ". ".join(sentences[:n]) + "."

    def _web_search(self, query):
        return f"[Search results for '{query}'] In production, integrate with SerpAPI or Tavily."

    def _fact_checker(self, claim):
        return f"[Fact check for '{claim}'] Recommend cross-referencing authoritative sources."


class AgentMemory:
    """Manages conversation history for the agent."""

    def __init__(self, max_entries=50):
        self.max_entries = max_entries
        self._entries = []

    def add(self, role, content):
        self._entries.append({"role": role, "content": content})
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries:]

    def get_context(self, n=5):
        recent = self._entries[-n:]
        return "\n".join(f"[{e['role'].upper()}]: {e['content']}" for e in recent)

    def clear(self):
        self._entries = []


class ResearchAgent:
    """
    Multi-step reasoning agent implementing ReAct pattern.
    Uses tools to answer complex queries with full reasoning trace.
    """

    def __init__(self, max_steps=5):
        self.max_steps = max_steps
        self.tools = ToolRegistry()
        self.memory = AgentMemory()
        logger.info("ResearchAgent initialized with max_steps=%d", max_steps)

    def run(self, query):
        logger.info("Agent starting for query: '%s'", query[:80])
        self.memory.add("user", query)
        steps = []

        for step_num in range(1, self.max_steps + 1):
            thought, action, tool_name, tool_input, is_final, final_answer = (
                self._simulate_reasoning(query, step_num)
            )

            step = AgentStep(
                step_number=step_num,
                thought=thought,
                action=action,
                is_final=is_final,
            )

            if is_final:
                steps.append(step)
                self.memory.add("agent", final_answer)
                logger.info("Agent completed at step %d", step_num)
                return AgentResult(
                    query=query,
                    final_answer=final_answer,
                    steps_taken=steps,
                    total_steps=step_num,
                    success=True,
                )

            if tool_name:
                tool_output = self.tools.invoke(tool_name, tool_input)
                step.tool_name = tool_name
                step.tool_output = tool_output
                self.memory.add("tool", f"{tool_name}: {tool_output}")

            steps.append(step)

        return AgentResult(
            query=query,
            final_answer="Max steps reached without completing task.",
            steps_taken=steps,
            total_steps=self.max_steps,
            success=False,
            error="max_steps_exceeded",
        )

    def _simulate_reasoning(self, query, step):
        if step == 1:
            return ("Searching for information.", "use_tool", "web_search", query, False, "")
        elif step == 2:
            return ("Summarizing findings.", "use_tool", "summarizer", f"Findings about: {query}", False, "")