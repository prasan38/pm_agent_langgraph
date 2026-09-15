from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

from graph_state.state import PMAgentState

CLARIFY_MODEL = "google_genai:gemini-3.6-flash"

SYSTEM_PROMPT = """You are a senior product manager doing intake on a new \
  feature request. Your job is ONLY to decide whether you have enough \
  information to write a decision record and user stories, and if not, what \
  to ask.
  
  Ground every question in the request and the knowledge base excerpts you're \
  given — don't ask about things already answered or already covered by the \
  docs. Prefer a small number of sharp, decision-relevant questions (scope, \
  constraints, success criteria, edge cases) over an exhaustive list. If the \
  request is already clear and unambiguous enough to act on, say so."""


USER_PROMPT_TEMPLATE = """Original request:
  {initial_request}
  
  Relevant knowledge base excerpts:
  {kb_text}
  
  Already answered:
  {answered_text}
  
  Do you have enough information to proceed, or do you need to ask more \
  questions?"""


class ClarifyOutput(BaseModel):
  """Structured verdict the clarify node needs each turn."""

  needs_more_input: bool = Field(
    description="True if there are still open questions blocking synthesis.",
  )

  questions: list[str] = Field(
    default_factory=list,
    description="New, non-redundant questions to ask. Empty if needs_more_input is False.",
  )

  confidence: float = Field(
    ge=0.0,
    le=1.0,
    description="0-1 confidence that, if unanswered, synthesis could still proceed reasonably.",
  )


def clarify_node(state: PMAgentState) -> dict:
   """
    LangGraph node: reads `initial_request`, `kb_context`, and any prior
    `answers`, and asks the LLM to decide whether enough is known to move
    to synthesis — and if not, what to ask next.
 
    Does NOT talk to the user directly. It only produces `open_questions`;
    the next node (human-in-the-loop, step 5) is responsible for actually
    surfacing those questions via `interrupt()` and collecting answers.
 
    Returns updates to `open_questions`, `clarify_turns`, and `status`.
    `status` becomes "ready_to_write" once the model reports no more
    questions are needed, or "clarifying" otherwise — this is what the
    step-6 conditional edge will branch on.
    """

   initial_request = state.get("initial_request", "")
   if not initial_request:
      raise ValueError("clarify_node requires state['initial_request] to be set.")

   kb_chunks = state.get("kb_context", [])
   kb_text = "\n\n---\n\n".join(kb_chunks) if kb_chunks else "(no knowledge loaded)"

   answers = state.get("answers", {})
   answered_text = (
      "\n".join(f"- Q: {q}\n A: {a}" for q, a in answers.items())
      if answers 
      else "(none yet)"
   )

   llm = init_chat_model(CLARIFY_MODEL, temperature=0)
   structured_llm = llm.with_structured_output(ClarifyOutput).with_retry(
      stop_after_attempt=4,
      wait_exponential_jitter=True,
   )

   result: ClarifyOutput = structured_llm.invoke(
      [
         {"role": "system", "content": SYSTEM_PROMPT},
         {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
               initial_request=initial_request,
               kb_text=kb_text,
               answered_text=answered_text
            ),
         },
      ]
   )

   new_questions = [q for q in result.questions if q not in answers]

   turns = state.get("clarify_turns", 0) + 1
   still_needs_input = result.needs_more_input and bool(new_questions)

   return {
      "open_questions": new_questions,
      "clarify_turns": turns,
      "status": "clarifying" if still_needs_input else "ready_to_write",
   }