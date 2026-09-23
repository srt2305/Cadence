from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from cadence.config import chat
from cadence.normalise import unwrap
from cadence.policy import Policy
from cadence.tools import CallLog, SchedulingBackend, make_tools
from cadence.triage import screen

ESCALATION_LINE = (
    "That needs a clinician now, not an appointment. "
    "I'm transferring you to our nurse line straight away and passing on what you've told me. "
    "If it worsens before they pick up, please call emergency services."
)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    escalated: bool
    escalation_reason: str


def build(
    policy: Policy, backend: SchedulingBackend, log: CallLog, screen_with_model: bool = True
) -> CompiledStateGraph:
    tools = make_tools(backend, log)
    model = chat(tools=tools)

    def triage(state: State) -> dict:
        last = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None
        )
        if last is None:
            return {}
        hit = screen(str(last.content), use_model=screen_with_model)
        if hit:
            return {"escalated": True, "escalation_reason": hit}
        return {}

    def agent(state: State) -> dict:
        system = SystemMessage(content=policy.render())
        reply = model.invoke([system] + state["messages"])
        if reply.content and not getattr(reply, "tool_calls", None):
            reply.content = unwrap(reply.content)
        return {"messages": [reply]}

    def escalate(state: State) -> dict:
        return {"messages": [AIMessage(content=ESCALATION_LINE)]}

    def after_triage(state: State) -> str:
        return "escalate" if state.get("escalated") else "agent"

    def after_agent(state: State) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else "end"

    g = StateGraph(State)
    g.add_node("triage", triage)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(tools))
    g.add_node("escalate", escalate)

    g.add_edge(START, "triage")
    g.add_conditional_edges("triage", after_triage, {"escalate": "escalate", "agent": "agent"})
    g.add_conditional_edges("agent", after_agent, {"tools": "tools", "end": END})
    g.add_edge("tools", "agent")
    g.add_edge("escalate", END)

    return g.compile()
