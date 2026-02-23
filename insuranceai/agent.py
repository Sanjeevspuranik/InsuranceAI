import json
import operator
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict, Any, TypedDict, Annotated, Optional

from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

_ = load_dotenv()


class ToolCallLog(TypedDict):
    """Log entry for a tool call."""
    timestamp: str
    tool_call_id: str
    name: str
    args: Any
    content: str


class AgentState(TypedDict):
    """Agent state with conversation history."""
    messages: Annotated[List[AnyMessage], operator.add]


class InsuranceAgent:
    """
    Agent for InsuranceAI workflow:
    - Runs YOLO prediction
    - Converts results to JSON
    - Generates GPT report
    - Creates PDF
    - Visualizes segments
    """

    def __init__(
        self,
        model: BaseLanguageModel,
        tools: List[BaseTool],
        checkpointer: Any = None,
        system_prompt: str = "",
        log_tools: bool = True,
        log_dir: Optional[str] = "logs",
    ):
        self.system_prompt = system_prompt
        self.log_tools = log_tools

        if self.log_tools:
            self.log_path = Path(log_dir or "logs")
            self.log_path.mkdir(exist_ok=True)

        # Define workflow
        workflow = StateGraph(AgentState)
        workflow.add_node("process", self.process_request)
        workflow.add_node("execute", self.execute_tools)
        workflow.add_conditional_edges(
            "process", self.has_tool_calls, {True: "execute", False: END}
        )
        workflow.add_edge("execute", "process")
        workflow.set_entry_point("process")

        self.workflow = workflow.compile(checkpointer=checkpointer)
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)  # type:ignore

    def process_request(self, state: AgentState) -> Dict[str, List[AnyMessage]]:
        """Process request with LLM."""
        messages = state["messages"]
        if self.system_prompt:
            messages = [SystemMessage(content=self.system_prompt)] + messages
        response = self.model.invoke(messages)
        return {"messages": [response]}

    def has_tool_calls(self, state: AgentState) -> bool:
        """Check if LLM response contains tool calls."""
        response = state["messages"][-1]
        return len(response.tool_calls) > 0  # type:ignore

    def execute_tools(self, state: AgentState) -> Dict[str, List[ToolMessage]]:
        """Execute tool calls from LLM response."""
        tool_calls = state["messages"][-1].tool_calls  # type:ignore
        results = []

        for call in tool_calls:
            print(f"Executing tool: {call}")
            if call["name"] not in self.tools:
                result = "invalid tool, please retry"
            else:
                result = self.tools[call["name"]].invoke(call["args"])

            results.append(
                ToolMessage(
                    tool_call_id=call["id"],
                    name=call["name"],
                    args=call["args"],
                    content=str(result),
                )
            )

        self._save_tool_calls(results)
        print("Returning to model processing!")
        return {"messages": results}

    def _save_tool_calls(self, tool_calls: List[ToolMessage]) -> None:
        """Save tool calls to JSON log file."""
        if not self.log_tools:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.log_path / f"tool_calls_{timestamp}.json"

        logs: List[ToolCallLog] = []
        for call in tool_calls:
            log_entry = {
                "tool_call_id": call.tool_call_id,
                "name": call.name,
                "args": call.args,  # type:ignore
                "content": call.content,
                "timestamp": datetime.now().isoformat(),
            }
            logs.append(log_entry)  # type:ignore

        with open(filename, "w") as f:
            json.dump(logs, f, indent=4)
