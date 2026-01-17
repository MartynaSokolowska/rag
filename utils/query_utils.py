import json


def parse_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        return None


from datetime import datetime


def validate_date_filter(result_json):
    """
    Sprawdza, czy date_filter jest True i daty są poprawne.
    Zwraca True jeśli filtr jest aktywny i daty OK, False w przeciwnym razie.
    """
    if not result_json.get("date_filter"):
        return True

    date_from = result_json.get("date_from")
    date_to = result_json.get("date_to")

    try:
        if date_from is not None:
            dt_from = datetime.fromisoformat(date_from)
        if date_to is not None:
            dt_to = datetime.fromisoformat(date_to)
    except (ValueError, TypeError):
        return False

    if date_from is not None and date_to is not None:
        if dt_from > dt_to:
            return False

    return True
