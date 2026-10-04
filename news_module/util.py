import re
import uuid 

def cuid() -> str:
    try:
        from cuid2 import cuid_wrapper
        return cuid_wrapper()()
    except ImportError:
        return str(uuid.uuid4())

def slugify(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text[:120] or "untitled"
