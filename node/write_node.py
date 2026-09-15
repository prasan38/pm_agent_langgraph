import os
from datetime import date

from graph_state.state import PMAgentState

FEATURES_DIR = "features"


def _render_markdown(decision_record: dict, user_stories: list[dict]) -> str:
  lines = ["# Decision Record", ""]
  lines.append(decision_record.get("summary", ""))
  lines.append("")
  lines.append("## Context")
  lines.append(decision_record.get("context", ""))
  lines.append("")
  lines.append("## Decision")
  lines.append(decision_record.get("decision", ""))
  lines.append("")
  lines.append("# User Stories")
  for story in user_stories:
    lines.append("")
    lines.append(f"## {story.get('title', '')}")
    lines.append(
      f"As a {story.get('as_a', '')}, I want {story.get('i_want', '')}, "
      f"so that {story.get('so_that', '')}."
    )
    lines.append("")
    lines.append("Acceptance criteria:")
    for criterion in story.get("acceptance_criteria", []):
      lines.append(f"- {criterion}")

  return "\n".join(lines) + "\n"


def write_node(state: PMAgentState) -> dict:
  """
  LangGraph node: persists the synthesized decision record and user stories
  to a local markdown file at `features/feature_{date}.md`.
  """

  decision_record = state.get("decision_record", {})
  user_stories = state.get("user_stories", [])

  os.makedirs(FEATURES_DIR, exist_ok=True)
  file_path = os.path.join(FEATURES_DIR, f"feature_{date.today().isoformat()}.md")

  with open(file_path, "w") as f:
    f.write(_render_markdown(decision_record, user_stories))

  return {"status": "published"}
