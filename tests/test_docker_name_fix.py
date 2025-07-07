import re
import pytest

def sanitize_docker_image_name(name: str) -> str:
    if not name or not str(name).strip():
        return "swe-task"
    sanitized = re.sub(r'[^a-z0-9_-]', '-', str(name).lower().strip())
    if sanitized.startswith('-'):
        sanitized = 'swe' + sanitized
    sanitized = sanitized.rstrip('-').rstrip('_').lstrip('_')
    if not sanitized or sanitized == "swe":
        return "swe-task"
    if len(sanitized) > 50:
        sanitized = sanitized[:50]
    return sanitized

def test_sanitize_docker_image_name():
    test_cases = [
        ("", "swe-task"),
        (None, "swe-task"),
        ("   ", "swe-task"),
        ("kareldb-connection-1", "kareldb-connection-1"),
        ("My Project", "my-project"),
        ("test@example.com", "test-example-com"),
        ("-dockerfile1", "swe-dockerfile1"),
        ("-test", "swe-test"),
        ("---", "swe-task"),
        ("project@#$%^&*()", "project"),
        ("CamelCase", "camelcase"),
        ("snake_case", "snake_case"),
        ("kebab-case", "kebab-case"),
        ("a" * 100, "a" * 50),
        ("___test___", "test"),
    ]
    for input_name, expected in test_cases:
        result = sanitize_docker_image_name(input_name)
        assert result == expected, f"Input: {input_name!r} -> {result!r} (expected: {expected!r})"

def test_image_name_generation():
    problematic_task_id = ""
    sanitized_task_id = sanitize_docker_image_name(problematic_task_id)
    setup_dockerfile_num = 1
    image_name = f"{sanitized_task_id}-dockerfile{setup_dockerfile_num}:latest"
    assert not image_name.startswith('-'), f"Image name should not start with hyphen: {image_name}" 