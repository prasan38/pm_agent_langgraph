from langgraph.types import interrupt

from graph_state.state import PMAgentState


def human_in_loop_node(state: PMAgentState) -> dict:
    """
    Surfaces each open_question to the user via interrupt(), pausing the
    graph until a human supplies an answer via Command(resume=...).
    """
    open_questions = state.get("open_questions", [])
    answers = dict(state.get("answers", {}))

    for question in open_questions:
        if question in answers:
            continue
        answer = interrupt({"question": question})
        answers[question] = answer

    return {
        "answers": answers,
        "open_questions": [],
    }
