import os

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph

from node.clarify_node import clarify_node
from node.human_in_loop_node import human_in_loop_node
from node.input_capture import capture_input_node
from node.kb_loader import load_kb_node
from graph_state.state import PMAgentState
from node.synthesize_node import synthesize_node
from node.write_node import write_node


def route_after_clarify(state: PMAgentState) -> str:
    return "human_in_loop" if state["status"] == "clarifying" else "synthesize"


builder = StateGraph(PMAgentState)
builder.add_node("capture_input", capture_input_node)
builder.add_node("load_kb", load_kb_node)
builder.add_node("clarify", clarify_node)
builder.add_node("human_in_loop", human_in_loop_node)
builder.add_node("synthesize", synthesize_node)
builder.add_node("write", write_node)

builder.add_edge(START, "capture_input")
builder.add_edge("capture_input", "load_kb")
builder.add_edge("load_kb", "clarify")
builder.add_conditional_edges("clarify", route_after_clarify, ["human_in_loop", "synthesize"])
builder.add_edge("human_in_loop", "clarify")
builder.add_edge("synthesize", "write")
builder.add_edge("write", END)

DATABASE_URL = os.environ["DATABASE_URL"]

# Kept open for the life of the process: `graph` is a module-level singleton
# imported by langgraph.json and test_hitl.py, so the connection must outlive
# this module's import rather than being closed when a `with` block exits.
_checkpointer_cm = PostgresSaver.from_conn_string(DATABASE_URL)
checkpointer = _checkpointer_cm.__enter__()
checkpointer.setup()

graph = builder.compile(checkpointer=checkpointer)
