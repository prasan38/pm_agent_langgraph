import re
import requests

from graph_state.state import PMAgentState

GITHUB_API = "https://api.github.com"
DOC_EXTENSIONS = (".md", ".mdx", ".txt")
MAX_FILES = 40
MAX_CHARS_PER_FILE = 8000


def _parse_github_url(url: str) -> tuple[str, str, str, str]:
  """
  Accepts a GitHub repo or folder URL, e.g. :
    https://github.com/owner/repo
    https://github.com/owner/repo/tree/main/docs
  Returns (owner, repo, ref, path)
  """

  match = re.match(
        r"https://github\.com/([^/]+)/([^/]+?)(?:/tree/([^/]+)/(.*))?/?$",
        url.rstrip("/"),
  )

  if not match:
    raise ValueError(f"Couldn't parse GitHub URL: {url}")
  owner, repo, ref, path = match.groups()
  print(f"owner={owner} \n repo={repo} \n ref:{ref} \n path={path}")
  return owner, repo, ref or "main", path or ""


# Fetch DIR
def _fetch_dir(owner: str, repo: str, ref: str, path: str, depth: int = 0) -> list[str]:
  """Recursively walk a repo path via the GitHub Contents API, collecting doc files."""

  if depth > 3:
    return []

  url = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}"
  resp = requests.get(url, params={"ref": ref}, timeout=15)
  resp.raise_for_status()
  entries = resp.json()
  if isinstance(entries, dict):
    entries = [entries]

  chunks = []
  for entry in entries:
    if len(chunks) >= MAX_FILES:
      break
    if entry["type"] == "dir":
      chunks.extend(_fetch_dir(owner, repo, ref, entry["path"], depth + 1))
    elif entry["type"] == "file" and  entry["name"].lower().endswith(DOC_EXTENSIONS):
      file_resp = requests.get(entry["download_url"], timeout=15)
      file_resp.raise_for_status()
      content = file_resp.text[:MAX_CHARS_PER_FILE]
      chunks.append(f"# {entry['path']}\n\n{content}")

  return chunks



def load_kb_node(state: PMAgentState) -> dict:
  """
  LangGraph node: reads state ['kb_source'] (a GitHub URL), fetched whatever
  markdown/text docs live there, and returns them as kb_context.

  Kept deliberately simple for now: whole-file text, no chunking or embeddings.
  Good enough while the docs fit in the model's context window; swap in a vector store later if the doc set grows too large.
  """

  owner, repo, ref, path = _parse_github_url(state["kb_source"])
  try:
    kb_context = _fetch_dir(owner, repo, ref, path)
  except requests.HTTPError as e:
    # Unauthenticated GitHub API calls are rate-limited to 60/hour.
    # Surface this clearly rather than failing silently.
    kb_context = [f"KB load failed: {e}"]

  return {"kb_context": kb_context}