# PM Agent

A LangGraph-based "PM Agent" that takes a raw feature request plus a GitHub docs URL, loads relevant knowledge-base context from that repo, iteratively clarifies the request with an LLM (via human-in-the-loop Q&A), then synthesizes a decision record and user stories and writes them to disk.

## Setup

```bash
# activate the venv (already created at .venv)
source .venv/bin/activate

# install/update deps
pip install -r requirements.txt

# start Postgres (used by the LangGraph checkpointer)
docker compose up -d
```

Copy `.env.example` to `.env` and fill in:

- `GOOGLE_API_KEY` — Google GenAI credential (used by `CLARIFY_MODEL`/`SYNTHESIZE_MODEL`, currently `google_genai:gemini-3.6-flash`)
- `DATABASE_URL` — Postgres connection string for the `PostgresSaver` checkpointer (matches `docker-compose.yml`'s `postgres` service when running locally)

## Usage

```bash
# run the full graph end-to-end, answering clarifying questions via stdin
python test_hitl.py "<feature request>" <github-docs-url>

# e.g.
python test_hitl.py "We need bulk CSV export for reports" https://github.com/langchain-ai/langgraph/tree/main/docs/docs/concepts
```

`langgraph.json` points `graphs.pm_agent` at `./agent/graph.py:graph`, so `langgraph dev` / `langgraph up` also work against it directly.

There is no test suite or linter configured yet. `test_hitl.py` is the way to exercise the full pipeline locally; individual node functions in `node/` can also be called directly for isolated debugging.

## Architecture

State flows through a `PMAgentState` TypedDict (`graph_state/state.py`), the single shared object LangGraph merges node return-dicts into. Every node takes the full state and returns only the keys it updates. Key fields:

- `messages` — LangChain messages, merged via `add_messages` reducer (the graph's entry point expects the user's request as the last `HumanMessage`).
- `initial_request` / `kb_source` / `kb_context` — the request text and loaded docs.
- `open_questions` / `answers` / `clarify_turns` — the clarification loop's working state.
- `decision_record` / `user_stories` — the synthesized output.
- `status` — one of `gathering | clarifying | ready_to_write | published`; conditional edges branch on this.

`agent/graph.py` wires the following pipeline, compiled with a `PostgresSaver` checkpointer backed by `DATABASE_URL`:

1. **`capture_input_node`** (`node/input_capture.py`) — pulls request text out of the last message into `initial_request`. Entry point of the graph.
2. **`load_kb_node`** (`node/kb_loader.py`) — parses `kb_source` as a GitHub repo/tree URL, recursively walks it via the unauthenticated GitHub Contents API (depth-limited to 3, capped at 40 files / 8000 chars each), and collects `.md`/`.mdx`/`.txt` files whole. Unauthenticated GitHub API calls are rate-limited to 60/hour; failures are captured into `kb_context` as an error string rather than raising.
3. **`clarify_node`** (`node/clarify_node.py`) — talks to an LLM with structured output to decide `needs_more_input`, generate non-redundant `questions`, and report a `confidence` score (currently unused in branching). Sets `status` to `ready_to_write` once the model has no more (non-duplicate) questions, else `clarifying`.
4. **`human_in_loop_node`** (`node/human_in_loop_node.py`) — reached when `status` is `clarifying`. Surfaces each unanswered entry in `open_questions` via `interrupt()`, pausing the graph until resumed with `Command(resume=<answer>)`; merges answers into `answers` and clears `open_questions`. Loops back to `clarify_node`, which may ask follow-ups or move on.
5. **`synthesize_node`** (`node/synthesize_node.py`) — reached once `status` is `ready_to_write`. Asks the LLM for a structured `DecisionRecord` and `list[UserStory]` from `initial_request`, `kb_context`, and `answers`. Sets `status` to `published`.
6. **`write_node`** (`node/write_node.py`) — renders the decision record and user stories to markdown and writes them to `features/feature_<date>.md`.

## Future Development

**Knowledge & Retrieval**
- Vector store for KB ingestion, replacing the current whole-file `load_kb_node` approach, for faster, more token-efficient retrieval that scales past the current 40-file/8000-char caps.

**Chat Integrations**
- Microsoft Teams bot integration, so users can launch the agent and answer clarifying questions (HITL) from a Teams chat instead of stdin.

**Output Destinations**
- Confluence publishing, writing the decision record and user stories directly to a Confluence page instead of (or alongside) the local `features/feature_<date>.md` file.

**Observability & Evals**
- Structured logging/tracing (e.g. LangSmith) across graph nodes.
- An eval suite for clarify-question quality and decision-record/user-story quality.

**Auth & Multi-tenancy**
- Authentication for chat surfaces (Teams bot) and Confluence writes.
- Secrets management beyond local `.env` files.
- Per-team/workspace isolation for concurrent users.

See [CLAUDE.md](CLAUDE.md) for more detail.
