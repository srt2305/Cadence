import json


def unwrap(content: object) -> str:
    """Return the model's plain text.

    Some models wrap replies in a JSON envelope such as {"answer": "..."} even
    when none was requested. Stripped here rather than asked for in a prompt.
    """
    text = str(content).strip()
    if not (text.startswith("{") and text.endswith("}")):
        return text
    try:
        data = json.loads(text)
    except ValueError:
        return text
    if isinstance(data, dict):
        for key in ("answer", "response", "content", "text", "message", "reply"):
            if isinstance(data.get(key), str):
                return data[key].strip()
    return text
