import subprocess
import os
import json
import runpy
import sys

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState, StateGraph

import issue_agent as agent


def test_logging_handles_unicode_on_legacy_windows_console(tmp_path):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "cp1252"
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(agent.__file__))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            'import logging; import agent; logging.getLogger("AgenticAI").info("Unicode: ↓")',
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr.decode("cp1252", errors="replace")
    assert b"Unicode: \\u2193" in result.stdout


def test_langsmith_tracing_is_opt_in(monkeypatch):
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    assert agent.configure_langsmith_tracing() is False
    assert "LANGSMITH_PROJECT" not in os.environ


def test_langsmith_tracing_uses_configured_project(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_PROJECT", "custom-agent-project")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-only-key")

    assert agent.configure_langsmith_tracing() is True
    assert os.environ["LANGSMITH_PROJECT"] == "custom-agent-project"


def initialize_repository(path):
    subprocess.run(["git", "init", str(path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test User"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.com"],
        check=True,
        capture_output=True,
    )
    (path / "tracked.txt").write_text("initial", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(path), "add", "--", "tracked.txt"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "Initial commit"],
        check=True,
        capture_output=True,
    )


def test_fetch_issue_details_reports_missing_issue(monkeypatch, tmp_path):
    monkeypatch.setattr(agent.STATE, "repo_root", tmp_path)

    def missing_issue(*args, **kwargs):
        raise subprocess.CalledProcessError(
            1, ["gh"], stderr="Could not resolve to an Issue with the number of 999."
        )

    monkeypatch.setattr(agent.subprocess, "run", missing_issue)

    with pytest.raises(agent.GitHubIssueNotFoundError, match="does not exist"):
        agent._fetch_issue_details(999)


def test_fetch_issue_details_distinguishes_missing_repository(monkeypatch, tmp_path):
    monkeypatch.setattr(agent.STATE, "repo_root", tmp_path)

    def missing_repository(*args, **kwargs):
        raise subprocess.CalledProcessError(
            1, ["gh"], stderr="GraphQL: Could not resolve to a repository"
        )

    monkeypatch.setattr(agent.subprocess, "run", missing_repository)

    with pytest.raises(agent.GitHubRepositoryUnavailableError, match="repository"):
        agent._fetch_issue_details(123)


def test_prepare_repository_reports_missing_local_directory(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        agent._prepare_repository(str(tmp_path / "missing-repo"))


def test_prepare_repository_reports_directory_that_is_not_git(tmp_path):
    with pytest.raises(ValueError, match="not a Git repository"):
        agent._prepare_repository(str(tmp_path))


def test_resolve_repo_path_rejects_paths_outside_repository(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(agent.STATE, "repo_root", repo.resolve())

    assert agent._resolve_repo_path("inside.txt") == repo / "inside.txt"
    with pytest.raises(ValueError, match="outside the target repository"):
        agent._resolve_repo_path("../outside.txt")


def test_write_file_does_not_write_outside_repository(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(agent.STATE, "repo_root", repo.resolve())
    outside = tmp_path / "outside.txt"

    result = agent.write_file.invoke(
        {"path": str(outside), "content": "must not be written"}
    )

    assert result.startswith("Error writing to file")
    assert not outside.exists()


def test_write_file_rejects_replacement_that_drops_existing_functions(tmp_path, monkeypatch):
    initialize_repository(tmp_path)
    (tmp_path / "calc.py").write_text(
        "def add(a, b):\n    return a + b\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(agent.STATE, "repo_root", tmp_path.resolve())
    monkeypatch.setattr(agent.STATE, "issue_branch", "fix-issue-123")

    result = agent.write_file.invoke(
        {
            "path": "calc.py",
            "content": "def process_map(values):\n    return list(values.values())\n",
        }
    )

    assert "would remove existing definitions: add" in result
    assert "def add(a, b)" in (tmp_path / "calc.py").read_text(encoding="utf-8")
    assert "def process_map" not in (tmp_path / "calc.py").read_text(encoding="utf-8")


def test_omitted_definition_detection_accepts_additive_replacement():
    existing = "def add(a, b):\n    return a + b\n"
    replacement = (
        "def add(a, b):\n    return a + b\n\n"
        "def process_map(values):\n    return list(values.values())\n"
    )

    assert agent._omitted_python_definitions(existing, replacement) == []
    assert agent._omitted_python_definitions(
        existing, "def process_map(values):\n    return list(values.values())\n"
    ) == ["add"]


def test_write_file_skips_identical_content_without_creating_branch(tmp_path, monkeypatch):
    initialize_repository(tmp_path)
    monkeypatch.setattr(agent.STATE, "repo_root", tmp_path.resolve())
    monkeypatch.setattr(agent.STATE, "issue_branch", "fix-issue-123")
    monkeypatch.setattr(agent.STATE, "written_files", {})

    result = agent.write_file.invoke(
        {"path": "tracked.txt", "content": "initial"}
    )

    assert result.startswith("No changes made")
    assert agent.STATE.written_files == {}
    assert agent.write_file.args_schema.model_fields.keys() == {"path", "content"}
    branch = subprocess.run(
        ["git", "-C", str(tmp_path), "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert branch != "fix-issue-123"


def test_append_file_preserves_existing_code_and_adds_new_content(tmp_path, monkeypatch):
    initialize_repository(tmp_path)
    monkeypatch.setattr(agent.STATE, "repo_root", tmp_path.resolve())
    monkeypatch.setattr(agent.STATE, "issue_branch", "fix-issue-128")
    monkeypatch.setattr(agent.STATE, "written_files", {})
    monkeypatch.setattr(agent.STATE, "read_files", set())
    original = (tmp_path / "tracked.txt").read_text(encoding="utf-8")
    agent.read_file.invoke({"path": "tracked.txt"})

    result = agent.append_file.invoke(
        {"path": "tracked.txt", "content": "def added():\n    return 42"}
    )

    updated = (tmp_path / "tracked.txt").read_text(encoding="utf-8")
    assert result.startswith("Successfully appended")
    assert updated.startswith(original)
    assert "def added():" in updated
    assert (tmp_path / "tracked.txt").resolve() in agent.STATE.written_files
    assert agent.append_file.args_schema.model_fields.keys() == {"path", "content"}


def test_append_file_requires_reading_file_first(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "file.py").write_text("original = True\n", encoding="utf-8")
    monkeypatch.setattr(agent.STATE, "repo_root", repo.resolve())
    monkeypatch.setattr(agent.STATE, "read_files", set())

    result = agent.append_file.invoke(
        {"path": "file.py", "content": "new_value = 1"}
    )

    assert result.startswith("Error: Read file.py")
    assert (repo / "file.py").read_text(encoding="utf-8") == "original = True\n"


def test_read_file_accepts_model_path_argument(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "calc.py").write_text("value = 1", encoding="utf-8")
    monkeypatch.setattr(agent.STATE, "repo_root", repo.resolve())

    result = agent.read_file.invoke({"path": "calc.py"})

    assert "value = 1" in result
    assert agent.read_file.args_schema.model_fields.keys() == {"path"}


def test_list_files_uses_requested_repository_subdirectory(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "module.py").write_text("value = 1", encoding="utf-8")
    (repo / "readme.txt").write_text("readme", encoding="utf-8")
    monkeypatch.setattr(agent.STATE, "repo_root", repo.resolve())

    result = agent.list_files.invoke({"path": "src"})

    assert "src/module.py" in result
    assert "readme.txt" not in result


def test_create_pull_request_requires_full_test_success(monkeypatch, tmp_path):
    monkeypatch.setattr(agent.STATE, "repo_root", tmp_path.resolve())
    monkeypatch.setattr(agent.STATE, "full_tests_passed", False)
    monkeypatch.setattr(agent.STATE, "issue_number", 123)

    result = agent.create_pull_request.invoke(
        {
            "title": "Fix issue",
            "body": "Fixes #123",
        }
    )

    assert "full test suite successfully" in result
    assert agent.create_pull_request.args_schema.model_fields.keys() == {
        "title",
        "body",
    }


def test_create_pull_request_accepts_native_title_body_tool_call(monkeypatch, tmp_path):
    monkeypatch.setattr(agent.STATE, "repo_root", tmp_path.resolve())
    monkeypatch.setattr(agent.STATE, "full_tests_passed", False)
    monkeypatch.setattr(agent.STATE, "issue_number", 123)
    tool_node = ToolNode([agent.create_pull_request])
    call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "create_pull_request",
                "args": {
                    "title": "Fix issue #123",
                    "body": "Implements the requested change.",
                },
                "id": "create-pr-call",
                "type": "tool_call",
            }
        ],
    )

    graph = StateGraph(MessagesState)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("tools")
    result = graph.compile().invoke({"messages": [call]})

    assert "full test suite successfully" in result["messages"][-1].content


def test_no_change_decision_requires_repository_inspection(monkeypatch):
    monkeypatch.setattr(agent.STATE, "written_files", {})
    monkeypatch.setattr(agent.STATE, "read_files", set())
    monkeypatch.setattr(agent.STATE, "no_changes_reason", None)

    result = agent.finish_without_changes.invoke({"reason": "Tests pass."})
    assert result.startswith("Error: Inspect at least one repository file")
    assert agent.STATE.no_changes_reason is None


def test_no_change_tool_interrupts_langgraph_tool_node(tmp_path, monkeypatch):
    monkeypatch.setattr(agent.STATE, "repo_root", tmp_path.resolve())
    monkeypatch.setattr(agent.STATE, "written_files", {})
    monkeypatch.setattr(agent.STATE, "read_files", {tmp_path / "inspected.py"})
    monkeypatch.setattr(agent.STATE, "no_changes_reason", None)
    tool_node = ToolNode([agent.finish_without_changes])
    call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "finish_without_changes",
                "args": {"reason": "Issue already satisfied."},
                "id": "finish-call",
                "type": "tool_call",
            }
        ],
    )

    graph = StateGraph(MessagesState)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("tools")

    with pytest.raises(agent.AgentFinishedWithoutChanges):
        graph.compile().invoke({"messages": [call]})

    assert agent.STATE.no_changes_reason == "Issue already satisfied."


def test_extract_text_tool_call_from_model_prose():
    content = (
        "I will add the requested helper.\n"
        '{"name":"write_file","parameters":{"path":"calc.py",'
        '"content":"def values_from_map(items):\\n    return list(items.values())"}}'
    )

    result = agent._extract_text_tool_call(content)

    assert result == (
        "write_file",
        {
            "path": "calc.py",
            "content": "def values_from_map(items):\n    return list(items.values())",
        },
    )


def test_normalize_common_model_tool_argument_aliases():
    assert agent._normalize_tool_arguments(
        "write_file", {"file_path": "calc.py", "content": "value = 1"}, 6
    ) == {"path": "calc.py", "content": "value = 1"}
    assert agent._normalize_tool_arguments(
        "append_file", {"file_path": "calc.py", "content": "def f(): pass"}, 6
    ) == {"path": "calc.py", "content": "def f(): pass"}
    assert agent._normalize_tool_arguments(
        "create_pull_request",
        {"title": "Implement request", "body": "What changed"},
        6,
    ) == {
        "title": "Implement request",
        "body": "What changed",
    }
    assert agent._normalize_tool_arguments(
        "create_pull_request",
        {"pr_title": "Implement request", "pr_body": "What changed"},
        6,
    ) == {
        "title": "Implement request",
        "body": "What changed",
    }


def test_incomplete_append_request_reports_missing_content():
    missing = agent._missing_required_tool_arguments(
        agent.append_file, {"path": "calc.py"}
    )

    assert missing == ["content"]


def test_runner_executes_text_tool_request_and_resumes_agent(tmp_path, monkeypatch):
    initialize_repository(tmp_path)
    (tmp_path / "test_baseline.py").write_text(
        "def test_baseline():\n    assert True\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "--", "test_baseline.py"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-m", "Add baseline test"],
        check=True,
        capture_output=True,
    )
    monkeypatch.setattr(
        agent.runner,
        "_fetch_issue_details",
        lambda issue_number: {
            "title": "Add map values helper",
            "body": "Add a helper that returns mapping values.",
            "state": "OPEN",
        },
    )
    stream_calls = []

    class FakeAgent:
        def stream(self, input_data, **kwargs):
            stream_calls.append(input_data["messages"])
            if len(stream_calls) == 1:
                response = AIMessage(
                    content=(
                        "I will implement the request.\n"
                        '{"name":"write_file","parameters":{"path":"calc.py",'
                        '"content":"def add(a, b):\\n    return a + b\\n\\n'
                        'def values_from_map(items):\\n'
                        '    return list(items.values())\\n"}}'
                    )
                )
            else:
                assert any(
                    isinstance(message, HumanMessage)
                    and "was executed" in message.content
                    for message in input_data["messages"]
                )
                result = agent.run_tests.invoke({})
                assert "Exit Code: 0" in result
                agent.STATE.pr_created = True
                response = AIMessage(content="Tests pass and PR was created.")
            yield {"messages": [*input_data["messages"], response]}

    monkeypatch.setattr(agent.runner, "ChatOllama", lambda **kwargs: object())
    monkeypatch.setattr(agent.runner, "create_agent", lambda **kwargs: FakeAgent())

    result = agent.run_agentic_fix(127, str(tmp_path))

    assert result == 0
    assert len(stream_calls) == 2
    assert "def values_from_map" in (tmp_path / "calc.py").read_text(encoding="utf-8")
    assert agent.STATE.full_tests_passed is True


def test_runner_retries_incomplete_append_request_without_removing_existing_function(
    tmp_path, monkeypatch
):
    initialize_repository(tmp_path)
    (tmp_path / "calc.py").write_text(
        "def add(a, b):\n    return a + b\n", encoding="utf-8"
    )
    (tmp_path / "test_calc.py").write_text(
        "from calc import add\n\n"
        "def test_add():\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "--", "calc.py", "test_calc.py"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-m", "Add calculator tests"],
        check=True,
        capture_output=True,
    )
    monkeypatch.setattr(
        agent.runner,
        "_fetch_issue_details",
        lambda issue_number: {
            "title": "Add map value processing",
            "body": "Add process_map(mapping) returning its values.",
            "state": "OPEN",
        },
    )
    initial_source = (tmp_path / "calc.py").read_text(encoding="utf-8")

    class FakeAgent:
        def __init__(self):
            self.calls = 0

        def stream(self, input_data, **kwargs):
            self.calls += 1
            if self.calls == 1:
                agent.read_file.invoke({"path": "calc.py"})
                response = AIMessage(
                    content=json.dumps(
                        {
                            "name": "append_file",
                            "parameters": {"path": "calc.py"},
                        }
                    )
                )
            elif self.calls == 2:
                assert any(
                    isinstance(message, HumanMessage)
                    and "missing required argument(s): content" in message.content
                    for message in input_data["messages"]
                )
                assert (tmp_path / "calc.py").read_text(encoding="utf-8") == initial_source
                response = AIMessage(
                    content=json.dumps(
                        {
                            "name": "append_file",
                            "parameters": {
                                "path": "calc.py",
                                "content": (
                                    "def process_map(mapping):\n"
                                    "    return list(mapping.values())"
                                ),
                            },
                        }
                    )
                )
            elif self.calls == 3:
                assert "Successfully appended" in input_data["messages"][-1].content
                assert (tmp_path / "calc.py").read_text(encoding="utf-8").startswith(
                    initial_source
                )
                agent.read_file.invoke({"path": "test_calc.py"})
                response = AIMessage(
                    content=json.dumps(
                        {
                            "name": "append_file",
                            "parameters": {
                                "path": "test_calc.py",
                                "content": (
                                    "from calc import process_map\n\n"
                                    "def test_process_map():\n"
                                    "    assert process_map({'a': 1, 'b': 2}) == [1, 2]"
                                ),
                            },
                        }
                    )
                )
            else:
                assert "Successfully appended" in input_data["messages"][-1].content
                result = agent.run_tests.invoke({})
                assert "Exit Code: 0" in result
                agent.STATE.pr_created = True
                response = AIMessage(content="The tests pass.")
            yield {"messages": [*input_data["messages"], response]}

    fake_agent = FakeAgent()
    monkeypatch.setattr(agent.runner, "ChatOllama", lambda **kwargs: object())
    monkeypatch.setattr(agent.runner, "create_agent", lambda **kwargs: fake_agent)

    result = agent.run_agentic_fix(129, str(tmp_path))

    assert result == 0
    assert fake_agent.calls == 4
    source = (tmp_path / "calc.py").read_text(encoding="utf-8")
    assert "def add(a, b)" in source
    assert "def process_map(mapping)" in source
    tests = (tmp_path / "test_calc.py").read_text(encoding="utf-8")
    assert "def test_add" in tests
    assert "def test_process_map" in tests


def test_prepare_repository_validates_repo_without_switching_branch(tmp_path):
    initialize_repository(tmp_path)

    original_branch = subprocess.run(
        ["git", "-C", str(tmp_path), "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    root = agent._prepare_repository(str(tmp_path))

    assert root == tmp_path.resolve()
    branch = subprocess.run(
        ["git", "-C", str(root), "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert branch == original_branch


def test_checkout_issue_branch_creates_branch_only_when_requested(tmp_path):
    initialize_repository(tmp_path)

    agent._checkout_issue_branch(tmp_path.resolve(), "fix-issue-123")

    branch = subprocess.run(
        ["git", "-C", str(tmp_path), "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert branch == "fix-issue-123"


def test_prepare_repository_refuses_preexisting_changes(tmp_path):
    initialize_repository(tmp_path)
    (tmp_path / "unrelated.txt").write_text("keep me", encoding="utf-8")

    with pytest.raises(RuntimeError, match="pre-existing changes"):
        agent._prepare_repository(str(tmp_path))

    assert (tmp_path / "unrelated.txt").read_text(encoding="utf-8") == "keep me"
    branch = subprocess.run(
        ["git", "-C", str(tmp_path), "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert branch != "fix-issue-123"


def test_agent_analyzes_green_baseline_and_stops_without_pr_when_issue_is_satisfied(tmp_path, monkeypatch):
    initialize_repository(tmp_path)
    (tmp_path / "test_baseline.py").write_text(
        "def test_baseline_passes():\n    assert 1 == 1\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "--", "test_baseline.py"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-m", "Add passing test"],
        check=True,
        capture_output=True,
    )

    monkeypatch.setattr(
        agent.runner,
        "_fetch_issue_details",
        lambda issue_number: {
            "title": "Add requested feature",
            "body": "Add behavior not covered by current tests",
            "state": "OPEN",
        },
    )

    class FakeAgent:
        def stream(self, input_data, **kwargs):
            assert "Baseline test suite passed: True" in input_data["messages"][0][1]
            agent.read_file.invoke({"path": "tracked.txt"})
            agent.finish_without_changes.invoke(
                {"reason": "Current behavior already satisfies the issue."}
            )

    monkeypatch.setattr(agent.runner, "ChatOllama", lambda **kwargs: object())
    monkeypatch.setattr(agent.runner, "create_agent", lambda **kwargs: FakeAgent())

    result = agent.run_agentic_fix(123, str(tmp_path))

    assert result == 0
    assert agent.STATE.pr_created is False
    branch = subprocess.run(
        ["git", "-C", str(tmp_path), "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert branch != "fix-issue-123"


def test_green_baseline_does_not_block_requirement_and_test_updates(tmp_path, monkeypatch):
    initialize_repository(tmp_path)
    (tmp_path / "calculator.py").write_text(
        "def add(a, b):\n    return a + b\n", encoding="utf-8"
    )
    (tmp_path / "test_calculator.py").write_text(
        "from calculator import add\n\n"
        "def test_add():\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "--", "calculator.py", "test_calculator.py"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-m", "Add calculator baseline"],
        check=True,
        capture_output=True,
    )
    monkeypatch.setattr(
        agent.runner,
        "_fetch_issue_details",
        lambda issue_number: {
            "title": "Add subtraction support",
            "body": "Add subtract(a, b) and test its behavior.",
            "state": "OPEN",
        },
    )

    class FakeAgent:
        def stream(self, input_data, **kwargs):
            assert "Baseline test suite passed: True" in input_data["messages"][0][1]
            agent.write_file.invoke(
                {
                    "path": "calculator.py",
                    "content": (
                        "def add(a, b):\n    return a + b\n\n"
                        "def subtract(a, b):\n    return a - b\n"
                    ),
                }
            )
            agent.write_file.invoke(
                {
                    "path": "test_calculator.py",
                    "content": (
                        "from calculator import add, subtract\n\n"
                        "def test_add():\n    assert add(1, 2) == 3\n\n"
                        "def test_subtract():\n    assert subtract(5, 3) == 2\n"
                    ),
                }
            )
            sys.path.insert(0, str(tmp_path))
            try:
                test_namespace = runpy.run_path(str(tmp_path / "test_calculator.py"))
                test_namespace["test_add"]()
                test_namespace["test_subtract"]()
            finally:
                sys.path.remove(str(tmp_path))
            agent.STATE.full_tests_passed = True
            agent.STATE.pr_created = True
            return iter(())

    monkeypatch.setattr(agent.runner, "ChatOllama", lambda **kwargs: object())
    monkeypatch.setattr(agent.runner, "create_agent", lambda **kwargs: FakeAgent())

    result = agent.run_agentic_fix(126, str(tmp_path))

    assert result == 0
    assert "def subtract(a, b)" in (tmp_path / "calculator.py").read_text(encoding="utf-8")
    assert "def test_subtract" in (tmp_path / "test_calculator.py").read_text(encoding="utf-8")
    branch = subprocess.run(
        ["git", "-C", str(tmp_path), "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert branch == "fix-issue-126"


def test_agent_stops_before_tests_when_issue_does_not_exist(tmp_path, monkeypatch, caplog):
    initialize_repository(tmp_path)
    monkeypatch.setattr(
        agent.runner,
        "_fetch_issue_details",
        lambda issue_number: (_ for _ in ()).throw(
            agent.GitHubIssueNotFoundError("Issue does not exist")
        ),
    )
    result = agent.run_agentic_fix(999, str(tmp_path))

    assert result == 1
    assert "[TOOL: run_tests]" not in caplog.text
    branch = subprocess.run(
        ["git", "-C", str(tmp_path), "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert branch != "fix-issue-999"


def test_agent_analyzes_issue_when_baseline_tests_fail_without_creating_empty_branch(tmp_path, monkeypatch):
    initialize_repository(tmp_path)
    (tmp_path / "test_baseline.py").write_text(
        "def test_baseline_fails():\n    assert 1 == 2\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "--", "test_baseline.py"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-m", "Add failing test"],
        check=True,
        capture_output=True,
    )
    monkeypatch.setattr(
        agent.runner,
        "_fetch_issue_details",
        lambda issue_number: {
            "title": "Fix broken behavior",
            "body": "Correct the output of the function.",
            "state": "OPEN",
        },
    )

    class FakeAgent:
        def stream(self, input_data, **kwargs):
            assert "Baseline test suite passed: False" in input_data["messages"][0][1]
            return iter(())

    monkeypatch.setattr(agent.runner, "ChatOllama", lambda **kwargs: object())
    monkeypatch.setattr(agent.runner, "create_agent", lambda **kwargs: FakeAgent())

    result = agent.run_agentic_fix(124, str(tmp_path))

    assert result == 1
    branch = subprocess.run(
        ["git", "-C", str(tmp_path), "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert branch != "fix-issue-124"


def test_write_file_creates_issue_branch_only_when_editing(tmp_path, monkeypatch):
    initialize_repository(tmp_path)
    monkeypatch.setattr(agent.STATE, "repo_root", tmp_path.resolve())
    monkeypatch.setattr(agent.STATE, "issue_branch", "fix-issue-125")
    monkeypatch.setattr(agent.STATE, "written_files", {})

    result = agent.write_file.invoke(
        {"path": "tracked.txt", "content": "updated"}
    )

    assert result.startswith("Successfully updated")
    branch = subprocess.run(
        ["git", "-C", str(tmp_path), "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert branch == "fix-issue-125"
