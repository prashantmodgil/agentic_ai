# GitHub Issue Fixing Agent

A command-line agent that takes a GitHub issue and a local Git repository, inspects the issue and code, updates implementation and tests when needed, runs the test suite, and can push a branch and open a pull request.

The agent uses an LLM through Ollama and LangChain/LangGraph tools. It does not currently include a web UI or a RAG index; it reads the current repository files directly.

## Architecture

```text
agent.py / python -m issue_agent
    |
    v
issue_agent.cli -> issue_agent.runner
                        |
                        +--> repository.py (validate local repo, fetch issue, manage branch)
                        +--> tools.py (repository edits, tests, no-change and PR actions)
                        +--> tool_calls.py (parse and validate model-emitted tool requests)
                        +--> state.py (per-run mutable workflow state and exceptions)
                        +--> config.py (logging and model/tool configuration)
```

### Main components

- `agent.py` — Backward-compatible CLI launcher.
- `issue_agent/cli.py` — Parses command-line arguments and starts a run.
- `issue_agent/runner.py` — Coordinates issue analysis, baseline tests, model calls, and workflow termination.
- `issue_agent/repository.py` — Validates the local repository and handles GitHub issue and branch operations.
- `issue_agent/tools.py` — Defines the LangChain tools for inspecting/editing files, running tests, and creating PRs.
- `issue_agent/tool_calls.py` — Normalizes and validates JSON-formatted tool requests from the model.
- `issue_agent/state.py` — Holds mutable state for the active run.
- `issue_agent/config.py` — Configures logging and agent settings.
- `test_agent.py` — regression tests for repository validation, tool arguments, issue handling, no-change termination, and the workflow.
- `.vscode/launch.json` — VS Code launch configuration for issue `1` and `./sample_repo`.
- `requirements.txt` — Python dependencies.

The agent project and `sample_repo/` are independent Git repositories. The agent project does not track the sample repository's files or Git history. You can point the agent at any other local Git repository by supplying its path.

## Requirements

- Python 3.10 or newer.
- Git.
- [GitHub CLI](https://cli.github.com/) installed and authenticated with access to the target repository.
- [Ollama](https://ollama.com/) installed and running, with the `llama3.1` model available.
- A local clone of the GitHub repository to work on.

The GitHub CLI identifies the issue using the target repository's Git context. The issue number must exist in that repository, and the authenticated GitHub account must have access to it. To create a PR, the target repository must also have a writable `origin` remote.

## Setup (Windows PowerShell)

From the project directory:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
ollama pull llama3.1
gh auth login
```

If PowerShell blocks virtual-environment activation, invoke its Python directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Check GitHub CLI authentication:

```powershell
gh auth status
```

Start Ollama if it is not already running, then run the agent:

```powershell
python agent.py <ISSUE_NUMBER> "<PATH_TO_LOCAL_REPOSITORY>"
```

Example using the separate sample repository:

```powershell
python agent.py 1 ".\sample_repo"
```

The package entry point is also available:

```powershell
python -m issue_agent 1 ".\sample_repo"
```

To work on a different project, provide the path to its local Git clone instead. The target repository must have a clean worktree before a run:

```powershell
python agent.py 42 "C:\path\to\another-repository"
```

The issue is fetched from that repository's GitHub context. If the issue requires changes and tests pass, the agent can commit and push a branch and open a real pull request in that repository. Use a disposable test repository if you do not intend to publish changes.

The VS Code configuration can also be started from **Run and Debug**. Change its issue number and repository path in `.vscode/launch.json` before use.

## Workflow and outcomes

1. The agent validates that the supplied path exists, is the Git repository root, and has no staged, modified, or untracked changes. It does not discard pre-existing changes.
2. It fetches the issue from GitHub. Missing/inaccessible local repositories and GitHub issues are reported and stop the run.
3. It runs the repository's baseline pytest suite. Passing baseline tests do **not** imply that an issue asking for new or changed behavior is already satisfied.
4. The LLM inspects the issue, relevant files, and test results. It may determine that the issue is already satisfied; that decision stops the run without edits or a PR. If the model emits a JSON tool request as text instead of a native tool call, the runner validates and executes the recognized tool request, then resumes the agent with its result.
5. For an actionable issue, the agent edits the implementation and relevant tests. Additive requirements use `append_file`, preserving existing code and tests; `write_file` is for required changes to existing behavior. It creates or switches to `fix-issue-<ISSUE_NUMBER>` when it makes the first actual edit.
6. The full pytest suite must pass after the latest edit before a PR can be created.
7. The PR tool stages only the files written by the agent in this run, commits them, pushes the issue branch to `origin`, and uses `gh` to open a PR.

The agent limits its tool calls to reduce the risk of an LLM/tool loop. A run that makes no edits exits without creating a PR only after an explicit no-change decision; an incomplete model response is reported as a failed run rather than as success.

## Tests

Run the agent's regression suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q test_agent.py
```

Run the separate sample repository's tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q .\sample_repo
```

## Safety notes and limitations

- The working tree must be clean before the agent starts. Commit, stash, or separately preserve local work first.
- The agent can write files and run tests in the supplied repository. Review the changes before using it on important repositories.
- Creating a PR involves Git commit/push operations and GitHub CLI access; there is no separate approval UI or dry-run mode yet.
- A single configured model (`llama3.1`) is used. Model output quality affects how well issue requirements are interpreted and implemented.
- This is a CLI workflow, not a UI application. Repository retrieval uses live file reads rather than RAG; there is no vector database or indexed repository snapshot.
- Tests are run with `pytest`. Repositories using another test runner or additional setup may need the test tool adapted.
