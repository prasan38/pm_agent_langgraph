import sys

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from agent.graph import graph


def main():
    request_text = sys.argv[1]
    kb_source = sys.argv[2]

    config = {"configurable": {"thread_id": "test-1"}}
    result = graph.invoke(
        {"messages": [HumanMessage(content=request_text)], "kb_source": kb_source},
        config,
    )

    while "__interrupt__" in result:
        question = result["__interrupt__"][0].value["question"]
        answer = input(f"\n{question}\n> ")
        result = graph.invoke(Command(resume=answer), config)

    print("\n--- decision record ---")
    print(result["decision_record"])

    print("\n--- user stories ---")
    for story in result["user_stories"]:
        print(story)

    print("\n--- status ---")
    print(result["status"])


if __name__ == "__main__":
    main()
