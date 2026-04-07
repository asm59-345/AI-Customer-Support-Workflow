"""
Upload all project files to Hugging Face Space using the HF Hub API.
More reliable than git push for HF Spaces.
"""
from huggingface_hub import HfApi
import os

# ── Configuration ──────────────────────────────────────────────────────────
HF_TOKEN   = os.environ.get("HF_TOKEN", "REPLACE_WITH_YOUR_HF_TOKEN")
REPO_ID    = "ashug85/AI-Customer-Support-Workflow"
REPO_TYPE  = "space"
LOCAL_DIR  = r"d:\Hackathon\2026\MetaXHuggingFace"

# Files/dirs to skip (don't upload to HF)
SKIP_PATTERNS = {
    ".git", ".claude", ".cursor", ".github", ".gemini", ".vibecheck",
    ".idea", ".vscode", "__pycache__", "venv", "env", ".venv",
    "workflows", "CLAUDE.md", "upload_to_hf.py",
    ".env", ".env.local",
}

SKIP_EXTENSIONS = {".pyc", ".pyo", ".db", ".sqlite", ".log", ".tmp", ".bak",
                   ".pt", ".bin", ".pth", ".onnx", ".pkl"}

# ── Upload ─────────────────────────────────────────────────────────────────
api = HfApi()
uploaded = []
skipped  = []

for root, dirs, files in os.walk(LOCAL_DIR):
    # Prune directories in-place so os.walk skips them
    dirs[:] = [d for d in dirs if d not in SKIP_PATTERNS and not d.startswith(".")]

    for filename in files:
        local_path = os.path.join(root, filename)
        rel_path   = os.path.relpath(local_path, LOCAL_DIR).replace("\\", "/")

        # Skip by pattern / extension
        parts = set(rel_path.split("/"))
        ext   = os.path.splitext(filename)[1].lower()

        if parts & SKIP_PATTERNS or ext in SKIP_EXTENSIONS:
            skipped.append(rel_path)
            continue

        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=rel_path,
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                token=HF_TOKEN,
            )
            print(f"  ✅ Uploaded: {rel_path}")
            uploaded.append(rel_path)
        except Exception as e:
            print(f"  ❌ FAILED:   {rel_path}  — {e}")

print(f"\n{'='*60}")
print(f"  Uploaded : {len(uploaded)} files")
print(f"  Skipped  : {len(skipped)} files")
print(f"\n  Space URL: https://huggingface.co/spaces/{REPO_ID}")
print(f"{'='*60}")
