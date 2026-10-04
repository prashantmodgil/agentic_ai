import json


def _message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block["text"]
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        )
    return ""

def _extract_text_tool_call(content: object) -> tuple[str, dict[str, object]] | None:
    text = _message_text(content)
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(candidate, dict):
            continue
        name = candidate.get("name")
        arguments = candidate.get("parameters", candidate.get("args"))
        if isinstance(name, str) and isinstance(arguments, dict):
            return name, arguments
    return None

def _normalize_tool_arguments(
    tool_name: str, arguments: dict[str, object], issue_number: int
) -> dict[str, object]:
    normalized = dict(arguments)
    if tool_name in {"read_file", "write_file", "append_file"} and "path" not in normalized:
        if "file_path" in normalized:
            normalized["path"] = normalized.pop("file_path")
    if tool_name == "run_tests" and "test_file" not in normalized:
        if "test_file_path" in normalized:
            normalized["test_file"] = normalized.pop("test_file_path")
    if tool_name == "create_pull_request":
        title = (
            normalized.get("title")
            or normalized.get("pr_title")
            or normalized.get("commit_message")
            or f"Fix issue #{issue_number}"
        )
        body = (
            normalized.get("body")
            or normalized.get("pr_body")
            or normalized.get("description")
            or f"Fixes #{issue_number}"
        )
        normalized = {"title": title, "body": body}
    return normalized

def _missing_required_tool_arguments(
    agent_tool: object, arguments: dict[str, object]
) -> list[str]:
    schema = getattr(agent_tool, "args_schema", None)
    fields = getattr(schema, "model_fields", {})
    return [
        field_name
        for field_name, field in fields.items()
        if field.is_required() and field_name not in arguments
    ]
