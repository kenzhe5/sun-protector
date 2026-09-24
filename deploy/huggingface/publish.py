"""Публикация Sun Protector на Hugging Face Spaces (Docker Space).

Что делает:
  1. Создаёт Space (если его ещё нет).
  2. Загружает нужные файлы проекта + Dockerfile и README для Spaces.
  3. Кладёт ключи из .env в секреты Space (значения нигде не печатаются).

Нужно в .env: HF_TOKEN (токен с правом write), OPENAI_API_KEY, CLAUDE_API_KEY,
по желанию LANGCHAIN_API_KEY.

Запуск из корня проекта: python deploy/huggingface/publish.py
"""
from __future__ import annotations

import os
import shutil
import tempfile

from dotenv import dotenv_values
from huggingface_hub import HfApi

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
SPACE_NAME = "sun-protector"

SECRETS = ["OPENAI_API_KEY", "CLAUDE_API_KEY", "LANGCHAIN_API_KEY"]
VARIABLES = {"LANGCHAIN_TRACING_V2": "true", "LANGCHAIN_PROJECT": "sun-protector"}
FOLDERS = ["agent-service", "mcp-server", "frontend"]
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", "chroma", ".env", ".DS_Store")


def main() -> None:
    env = {k: (v or "").strip().strip('"').strip("'") for k, v in dotenv_values(os.path.join(ROOT, ".env")).items()}
    token = env.get("HF_TOKEN")
    if not token:
        raise SystemExit("В .env нет HF_TOKEN — создайте токен (write) на huggingface.co/settings/tokens")

    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo_id = f"{user}/{SPACE_NAME}"
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    print("Space:", repo_id)

    # ключи — в секреты Space, до загрузки кода, чтобы первая сборка уже их видела
    for key in SECRETS:
        if env.get(key):
            api.add_space_secret(repo_id, key, env[key])
            print("  секрет добавлен:", key)
    for key, value in VARIABLES.items():
        api.add_space_variable(repo_id, key, value)

    with tempfile.TemporaryDirectory() as stage:
        for folder in FOLDERS:
            shutil.copytree(os.path.join(ROOT, folder), os.path.join(stage, folder), ignore=SKIP)
        shutil.copy(os.path.join(HERE, "Dockerfile"), os.path.join(stage, "Dockerfile"))
        shutil.copy(os.path.join(HERE, "README.md"), os.path.join(stage, "README.md"))
        api.upload_folder(
            repo_id=repo_id, repo_type="space", folder_path=stage,
            commit_message="Публикация Sun Protector", delete_patterns=["*"],
        )

    host = f"{user}-{SPACE_NAME}".lower().replace("_", "-")
    print(f"Готово. Сборка займёт несколько минут.\n  Страница: https://huggingface.co/spaces/{repo_id}\n  Приложение: https://{host}.hf.space")


if __name__ == "__main__":
    main()
