from langchain_core.messages import HumanMessage

from graph_state.state import PMAgentState

def capture_input_node(state: PMAgentState) -> dict:
  """
  Entry point node for the graph.

  Expects the graph to be invoked with the user's request already sitting in state['messages'], e.g,:
  graph.invoke({"messages": [HumanMessage(content="We need a way to ...")]})

  Copies that text into `initial_request` so every downstream node has one
  stable field to read from, instead of re-parsing message history each time.
  """

  messages = state.get("messages", [])
  if not messages:
    raise ValueError("capture_input_node expects at least one message in state['messages'].")

  last_message = messages[-1]
  request_text = getattr(last_message, "content", None)
  if request_text is None and isinstance(last_message, dict):
    request_text = last_message.get("content")

  if not request_text or not request_text.strip():
    raise ValueError("Couldn't extract request text from the last message.")

  return {
    "initial_request": request_text.strip(),
    "status": "gathering"
  }