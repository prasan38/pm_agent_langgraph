from typing import Annotated, Literal, TypedDict
from langgraph.graph.message import add_messages

class PMAgentState(TypedDict):
  """
  Shared state passed between every node in the graph. 
  Lang graph merges each node's turn dict into this state automatically, so a node only
  needs to return the keys it''s updating.
  """

  messages: Annotated[list, add_messages]

  # Input
  initial_request: str

  # --- Knowledge base ---
  kb_source: str

  # GitHub URL the user gave us for existing docs.
  kb_context: list[str]

  open_questions: list[str]

  answers: dict[str, str]

  clarify_turns: int

  # --- Output ---
  decision_record: dict

  user_stories: list[dict]

  status: Literal["gathering", "clarifying", "ready_to_write", "published"]