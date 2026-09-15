from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

from graph_state.state import PMAgentState

SYNTHESIZE_MODEL = "google_genai:gemini-3.6-flash"

SYSTEM_PROMPT = """You are a senior product manager writing up the outcome of \
  an intake conversation. You have a feature request, relevant knowledge base \
  excerpts, and answers to every clarifying question that was needed. Write a \
  concise decision record and a set of user stories that a team could start \
  building from.

  Keep the decision record short and decision-focused — summary, context, and \
  the decision itself. Keep user stories independent and testable, each with \
  clear acceptance criteria."""


USER_PROMPT_TEMPLATE = """Original request:
  {initial_request}

  Relevant knowledge base excerpts:
  {kb_text}

  Clarifying Q&A:
  {answered_text}

  Write the decision record and user stories."""


class UserStory(BaseModel):
  """One user story with acceptance criteria."""

  title: str = Field(description="Short title for the story.")
  as_a: str = Field(description="The user or role this story is written for.")
  i_want: str = Field(description="What the user wants to do.")
  so_that: str = Field(description="The benefit or goal behind the want.")
  acceptance_criteria: list[str] = Field(
    default_factory=list,
    description="Concrete, testable conditions for the story to be done.",
  )


class DecisionRecord(BaseModel):
  """Short decision record summarizing the intake conversation."""

  summary: str = Field(description="One or two sentence summary of the decision.")
  context: str = Field(description="Why this request came up and what's driving it.")
  decision: str = Field(description="What was decided to build, in concrete terms.")


class SynthesisOutput(BaseModel):
  """Structured output the synthesize node needs."""

  decision_record: DecisionRecord
  user_stories: list[UserStory] = Field(default_factory=list)


def synthesize_node(state: PMAgentState) -> dict:
  """
  LangGraph node: reads `initial_request`, `kb_context`, and `answers` once
  clarification is done, and asks the LLM to write a decision record and
  user stories from them.

  Returns updates to `decision_record`, `user_stories`, and `status`.
  `status` becomes "published" — the graph terminates here.
  """

  initial_request = state.get("initial_request", "")
  if not initial_request:
    raise ValueError("synthesize_node requires state['initial_request'] to be set.")

  kb_chunks = state.get("kb_context", [])
  kb_text = "\n\n---\n\n".join(kb_chunks) if kb_chunks else "(no knowledge loaded)"

  answers = state.get("answers", {})
  answered_text = (
    "\n".join(f"- Q: {q}\n A: {a}" for q, a in answers.items())
    if answers
    else "(no clarifying questions were needed)"
  )

  llm = init_chat_model(SYNTHESIZE_MODEL, temperature=0)
  structured_llm = llm.with_structured_output(SynthesisOutput).with_retry(
    stop_after_attempt=4,
    wait_exponential_jitter=True,
  )

  result: SynthesisOutput = structured_llm.invoke(
    [
      {"role": "system", "content": SYSTEM_PROMPT},
      {
        "role": "user",
        "content": USER_PROMPT_TEMPLATE.format(
          initial_request=initial_request,
          kb_text=kb_text,
          answered_text=answered_text,
        ),
      },
    ]
  )

  return {
    "decision_record": result.decision_record.model_dump(),
    "user_stories": [s.model_dump() for s in result.user_stories],
    "status": "published",
  }
