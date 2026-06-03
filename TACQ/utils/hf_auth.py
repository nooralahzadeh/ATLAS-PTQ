"""Shared HuggingFace authentication for TACQ entry points."""
import os

from dotenv import load_dotenv


def login_huggingface():
    load_dotenv()
    token = (
        os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
    )
    if not token:
        return False
    import huggingface_hub

    huggingface_hub.login(token=token)
    return True


def require_huggingface_token():
    if login_huggingface():
        return
    raise RuntimeError(
        "HuggingFace token required for meta-llama/Meta-Llama-3-8B-Instruct. "
        "Set HUGGINGFACE_TOKEN in TACQ/.env (accept the model license on HuggingFace first)."
    )
