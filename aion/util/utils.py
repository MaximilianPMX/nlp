import os

def validate_file_path(file_path: str, allowed_extensions: list = None) -> bool:
    """Validates if a file path exists and (optionally) has one of the allowed extensions."""
    if not os.path.exists(file_path):
        return False

    if allowed_extensions:
        _, ext = os.path.splitext(file_path)
        if ext.lower() not in allowed_extensions:
            return False

    return True


def validate_is_string(input_data: any) -> bool:
    """Validates if the given input is a string."""
    return isinstance(input_data, str)


def handle_error(message: str, exception: Exception = None) -> None:
    """Handles errors by printing a message and optionally re-raising the exception."""
    print(f"Error: {message}")
    if exception:
        raise exception