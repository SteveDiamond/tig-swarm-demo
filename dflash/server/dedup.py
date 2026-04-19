import hashlib
import re


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def fingerprint(title: str, tag: str) -> str:
    raw = f"{normalize(title)}|{tag}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
