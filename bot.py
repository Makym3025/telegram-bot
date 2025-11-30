import asyncio
import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (ApplicationBuilder, CommandHandler, ContextTypes,
                          MessageHandler, filters)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

WORKSPACE_ROOT = Path(os.environ.get("WORKSPACE_ROOT", Path.cwd())).resolve()
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-pro")


def ensure_workspace(path: Path) -> Path:
    path = path.resolve()
    if not str(path).startswith(str(WORKSPACE_ROOT)):
        raise ValueError("Path escapes workspace root")
    return path


@dataclass
class ExecutionResult:
    action: str
    detail: str
    output: str
    success: bool


class WorkspaceManager:
    def __init__(self, root: Path) -> None:
        self.root = ensure_workspace(root)

    def resolve(self, path_str: str) -> Path:
        path = ensure_workspace(self.root.joinpath(path_str).resolve())
        return path

    def write_file(self, path_str: str, content: str) -> ExecutionResult:
        path = self.resolve(path_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return ExecutionResult("write_file", str(path), "written", True)

    def append_file(self, path_str: str, content: str) -> ExecutionResult:
        path = self.resolve(path_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(content)
        return ExecutionResult("append_file", str(path), "appended", True)

    def read_file(self, path_str: str) -> ExecutionResult:
        path = self.resolve(path_str)
        text = path.read_text(encoding="utf-8")
        return ExecutionResult("read_file", str(path), text, True)


class CommandRunner:
    def __init__(self, timeout: int = 60) -> None:
        self.timeout = timeout

    def run(self, command: str) -> ExecutionResult:
        logger.info("Running command: %s", command)
        try:
            completed = subprocess.run(
                command,
                shell=True,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            output = completed.stdout + completed.stderr
            success = completed.returncode == 0
            detail = f"exit_code={completed.returncode}"
            return ExecutionResult("command", detail, output.strip(), success)
        except subprocess.TimeoutExpired as exc:
            return ExecutionResult("command", "timeout", str(exc), False)


@dataclass
class Session:
    api_key: Optional[str]
    history: List[Dict[str, str]] = field(default_factory=list)


class AIAgent:
    allowed_actions = {"command", "write_file", "append_file", "read_file"}

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL) -> None:
        self.api_key = api_key
        self.model_name = model
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    @property
    def system_prompt(self) -> str:
        return (
            "You are an autonomous coding agent working inside a sandboxed workspace. "
            "You must respond strictly in JSON with a top-level 'steps' list. Each item "
            "is an action with keys: action (command|write_file|append_file|read_file), "
            "input or path/content as needed. Prefer concise commands. Include only actions "
            "that are necessary to complete the user's request."
        )

    def build_prompt(self, task: str, history: List[Dict[str, str]]) -> List[str]:
        preamble = self.system_prompt
        context = "\n".join([f"user: {h['user']}\nassistant: {h['assistant']}" for h in history[-6:]])
        return [preamble, "Context:", context, "Task:", task]

    def plan_actions(self, task: str, history: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        prompt = self.build_prompt(task, history)
        response = self.model.generate_content(prompt)
        text = response.text
        logger.info("Model response: %s", text)
        data = json.loads(text)
        steps = data.get("steps", [])
        valid_steps = []
        for step in steps:
            action = step.get("action")
            if action not in self.allowed_actions:
                continue
            valid_steps.append(step)
        return valid_steps


async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привіт! Я AI-агент, що може планувати дії через Gemini: виконувати команди, "
        "створювати та читати файли. Використовуйте /task щоб поставити завдання."
    )


def get_session(context: ContextTypes.DEFAULT_TYPE, user_id: int) -> Session:
    sessions = context.bot_data.setdefault("sessions", {})
    if user_id not in sessions:
        sessions[user_id] = Session(api_key=os.environ.get("GEMINI_API_KEY"))
    return sessions[user_id]


async def handle_setkey(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if not message:
        return
    key = message.text.partition(" ")[2].strip()
    if not key:
        await message.reply_text("Надайте ключ після /setkey")
        return
    session = get_session(context, message.from_user.id)
    session.api_key = key
    await message.reply_text("API ключ збережено у поточній сесії")


async def handle_task(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if not message:
        return
    session = get_session(context, message.from_user.id)
    if not session.api_key:
        await message.reply_text("Спочатку надайте GEMINI_API_KEY через змінну середовища.")
        return

    task_text = message.text.partition(" ")[2].strip()
    if not task_text:
        await message.reply_text("Додайте опис завдання після /task")
        return

    agent = AIAgent(session.api_key)
    workspace = WorkspaceManager(WORKSPACE_ROOT)
    runner = CommandRunner()

    try:
        steps = await asyncio.to_thread(agent.plan_actions, task_text, session.history)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to plan actions")
        await message.reply_text(f"Помилка планування: {exc}")
        return

    results: List[ExecutionResult] = []
    for step in steps:
        action = step.get("action")
        if action == "command":
            res = runner.run(step.get("input", ""))
        elif action == "write_file":
            res = workspace.write_file(step.get("path", ""), step.get("content", ""))
        elif action == "append_file":
            res = workspace.append_file(step.get("path", ""), step.get("content", ""))
        elif action == "read_file":
            res = workspace.read_file(step.get("path", ""))
        else:
            res = ExecutionResult(action or "unknown", "skipped", "", False)
        results.append(res)

    session.history.append({"user": task_text, "assistant": json.dumps([r.__dict__ for r in results])})

    summary_lines = [f"*{r.action}*: {r.detail}\n``{r.output}``" for r in results]
    await message.reply_text("\n\n".join(summary_lines), parse_mode=ParseMode.MARKDOWN_V2)


async def handle_run(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if not message:
        return
    command = message.text.partition(" ")[2].strip()
    if not command:
        await message.reply_text("Надайте команду після /run")
        return
    runner = CommandRunner()
    result = runner.run(command)
    await message.reply_text(f"exit={result.detail}\n``{result.output}``", parse_mode=ParseMode.MARKDOWN_V2)


async def handle_read(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if not message:
        return
    path = message.text.partition(" ")[2].strip()
    if not path:
        await message.reply_text("Надайте шлях після /read")
        return
    workspace = WorkspaceManager(WORKSPACE_ROOT)
    try:
        result = workspace.read_file(path)
        await message.reply_text(f"``{result.output}``", parse_mode=ParseMode.MARKDOWN_V2)
    except Exception as exc:  # noqa: BLE001
        await message.reply_text(f"Помилка: {exc}")


async def handle_write(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if not message:
        return
    parts = message.text.split(maxsplit=2)
    if len(parts) < 3:
        await message.reply_text("Використовуйте /write <шлях> <текст>")
        return
    path = parts[1]
    content = parts[2]
    workspace = WorkspaceManager(WORKSPACE_ROOT)
    try:
        result = workspace.write_file(path, content)
        await message.reply_text(f"Файл записано: {result.detail}")
    except Exception as exc:  # noqa: BLE001
        await message.reply_text(f"Помилка: {exc}")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Спробуйте /task для запуску агента або /run для виконання команди.")


def build_app(token: str) -> Any:
    application = ApplicationBuilder().token(token).build()
    application.add_handler(CommandHandler("start", handle_start))
    application.add_handler(CommandHandler("setkey", handle_setkey))
    application.add_handler(CommandHandler("task", handle_task))
    application.add_handler(CommandHandler("run", handle_run))
    application.add_handler(CommandHandler("read", handle_read))
    application.add_handler(CommandHandler("write", handle_write))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    return application


async def main() -> None:
    token = os.environ.get("BOT_TOKEN")
    if not token:
        raise RuntimeError("BOT_TOKEN is not set")
    app = build_app(token)
    logger.info("Starting bot in workspace: %s", WORKSPACE_ROOT)
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    await app.updater.idle()


if __name__ == "__main__":
    asyncio.run(main())
