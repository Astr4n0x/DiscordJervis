"""Discord bot that answers any question with a Groq chat model."""

from __future__ import annotations

import asyncio
import os
import re
from collections import defaultdict

import discord
from discord import app_commands
from dotenv import load_dotenv
from groq import APIError, Groq

load_dotenv()

DEFAULT_MODEL = "openai/gpt-oss-120b"
DISCORD_LIMIT = 1900
SEARCH_MODELS = {
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-safeguard-20b",
}
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful Discord assistant. Answer the user's question directly, "
    "in a tone that fits a chat server. Keep replies focused unless the user asks for detail. "
    "Format with Discord markdown only: # ## and ### headings, -# subtext, **bold**, "
    "*italics*, __underline__, ~~strike~~, `code`, fenced code blocks, - lists, "
    "> quotes, and [label](url) links. Do not use markdown tables, HTML, or "
    "![image](url) syntax. "
    "When a question depends on news, prices, weather, sports, releases, or other "
    "live facts, search the web and include the source links you used. "
    "For coding help, opinions, and general knowledge, answer directly."
)
SKIP_MODEL_MARKERS = (
    "whisper",
    "orpheus",
    "prompt-guard",
    "tts",
    "playai",
    "safeguard",
)


def env_required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise SystemExit(
            f"Missing {name}. Copy .env.example to .env and fill it in."
        )
    return value


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer, got {raw!r}.") from exc


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise SystemExit(f"{name} must be true or false, got {raw!r}.")


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be a number, got {raw!r}.") from exc


_FENCE_SPLIT = re.compile(r"(```[\s\S]*?```)")
_HEADING = re.compile(r"^(#{1,6})\s*(.+?)\s*#*\s*$")
_SETEXT = re.compile(r"^(=+|-+)\s*$")
_HR = re.compile(r"^(?:-{3,}|\*{3,}|_{3,})\s*$")
_IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_ANCHOR = re.compile(
    r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
    re.IGNORECASE,
)
_HTML_TAG = re.compile(
    r"</?(?:br|p|div|span|h[1-6]|ul|ol|li|table|thead|tbody|tr|td|th|img|pre|hr|"
    r"blockquote|sup|sub|section|article)\b[^>]*>",
    re.IGNORECASE,
)


def _split_table_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _is_table_separator(line: str) -> bool:
    if "|" not in line:
        return False
    cells = _split_table_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def _is_table_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2


def _render_table(header: list[str], body: list[list[str]]) -> str:
    width = max([len(header), *(len(row) for row in body)] or [0])
    header = [*header, *([""] * (width - len(header)))]
    rows = [[*row, *([""] * (width - len(row)))] for row in body]
    if width == 2 and rows:
        lines = [f"**{header[0]}** — **{header[1]}**"]
        lines.extend(f"- {row[0]} — {row[1]}" for row in rows)
        return "\n".join(lines)

    def align(row: list[str], sizes: list[int]) -> str:
        return " | ".join(cell.ljust(sizes[index]) for index, cell in enumerate(row))

    sizes = [
        max(len(header[index]), *(len(row[index]) for row in rows))
        for index in range(width)
    ]
    rule = "-+-".join("-" * size for size in sizes)
    lines = [align(header, sizes), rule, *(align(row, sizes) for row in rows)]
    return "```\n" + "\n".join(lines) + "\n```"


def _html_to_discord(text: str) -> str:
    text = _ANCHOR.sub(lambda match: f"[{match.group(2)}]({match.group(1)})", text)
    text = re.sub(r"<(?:b|strong)\b[^>]*>", "**", text, flags=re.IGNORECASE)
    text = re.sub(r"</(?:b|strong)>", "**", text, flags=re.IGNORECASE)
    text = re.sub(r"<(?:i|em)\b[^>]*>", "*", text, flags=re.IGNORECASE)
    text = re.sub(r"</(?:i|em)>", "*", text, flags=re.IGNORECASE)
    text = re.sub(r"<(?:u)\b[^>]*>", "__", text, flags=re.IGNORECASE)
    text = re.sub(r"</u>", "__", text, flags=re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    return _HTML_TAG.sub("", text)


def _format_inline(text: str) -> str:
    pieces = re.split(r"(`[^`\n]*`)", text)
    formatted: list[str] = []
    for piece in pieces:
        if piece.startswith("`") and piece.endswith("`"):
            formatted.append(piece)
            continue
        piece = _IMAGE.sub(lambda match: match.group(2).strip(), piece)
        piece = _html_to_discord(piece)
        formatted.append(piece)
    return "".join(formatted)


def _format_prose(text: str) -> str:
    lines = text.splitlines()
    output: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        nxt = lines[index + 1] if index + 1 < len(lines) else ""
        if (
            line.strip()
            and not line.strip().startswith("|")
            and _SETEXT.match(nxt)
            and not _is_table_separator(nxt)
        ):
            level = "#" if nxt.strip().startswith("=") else "##"
            output.append(f"{level} {line.strip()}")
            index += 2
            continue
        if _is_table_row(line) and _is_table_separator(nxt):
            header = _split_table_row(line)
            index += 2
            body: list[list[str]] = []
            while index < len(lines) and _is_table_row(lines[index]):
                body.append(_split_table_row(lines[index]))
                index += 1
            output.append(_render_table(header, body))
            continue

        stripped = line.strip()
        if _HR.match(stripped):
            output.append("")
            index += 1
            continue

        heading = _HEADING.match(stripped)
        if heading and heading.group(2).strip():
            level = min(len(heading.group(1)), 3)
            title = heading.group(2).strip().strip("#").strip()
            prefix = "-#" if len(heading.group(1)) > 3 else "#" * level
            output.append(_format_inline(f"{prefix} {title}"))
            index += 1
            continue

        if re.match(r"^\s*-#", line):
            line = re.sub(r"^(\s*-#)(?=\S)", r"\1 ", line)
        elif re.match(r"^\s*(?:-(?!#)|\d+\.)(?=\S)", line):
            line = re.sub(r"^(\s*(?:-(?!#)|\d+\.))(?=\S)", r"\1 ", line)
        line = re.sub(r"^(\s*>{1,3})(?=\S)", r"\1 ", line)
        line = re.sub(r"^(\s*)[-*]\s+\[ \]\s+", r"\1- ", line)
        line = re.sub(r"^(\s*)[-*]\s+\[[xX]\]\s+", r"\1- ✅ ", line)
        output.append(_format_inline(line))
        index += 1
    rendered = "\n".join(output)
    if text.endswith("\n") and rendered and not rendered.endswith("\n"):
        rendered += "\n"
    return rendered


def to_discord_markdown(text: str) -> str:
    """Turn common Markdown into the subset Discord actually renders."""
    if not text:
        return ""
    rendered = [
        piece if piece.startswith("```") else _format_prose(piece)
        for piece in _FENCE_SPLIT.split(text)
    ]
    return re.sub(r"\n{3,}", "\n\n", "".join(rendered)).strip()


def chunk_text(text: str, limit: int = DISCORD_LIMIT) -> list[str]:
    """Split a reply into Discord-sized messages, keeping code fences intact."""
    text = to_discord_markdown(text) or "(The model returned an empty reply.)"
    if len(text) <= limit:
        return [text]

    parts: list[str] = []
    current: list[str] = []
    size = 0
    in_fence = False
    fence_line = "```"

    def flush() -> None:
        nonlocal current, size
        block = "\n".join(current).strip()
        if block:
            parts.append(block)
        current = []
        size = 0

    for line in text.splitlines() or [text]:
        extra = len(line) + 1
        reserve = 4 if in_fence else 0
        if current and size + extra + reserve > limit:
            if in_fence:
                current.append("```")
            flush()
            if in_fence:
                current.append(fence_line)
                size = len(fence_line) + 1
        if len(line) > limit:
            flush()
            start = 0
            while start < len(line):
                parts.append(line[start : start + limit])
                start += limit
            continue
        current.append(line)
        size += extra
        if line.strip().startswith("```"):
            if not in_fence:
                fence_line = line
            in_fence = not in_fence

    if in_fence:
        current.append("```")
    flush()
    return parts or ["(The model returned an empty reply.)"]


def is_chat_model(model_id: str) -> bool:
    lowered = model_id.lower()
    return not any(marker in lowered for marker in SKIP_MODEL_MARKERS)


def guild_id_from_env() -> int | None:
    raw = os.getenv("DISCORD_GUILD_ID", "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise SystemExit("DISCORD_GUILD_ID must be a numeric server id.") from exc


class GroqBot(discord.Client):
    def __init__(self) -> None:
        intents = discord.Intents.default()
        super().__init__(
            intents=intents,
            allowed_mentions=discord.AllowedMentions.none(),
        )
        self.tree = app_commands.CommandTree(self)
        self._groq: Groq | None = None
        self.model = os.getenv("GROQ_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
        self.system_prompt = os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT).strip()
        self.web_search = env_bool("WEB_SEARCH", True)
        self.max_tokens = env_int("GROQ_MAX_TOKENS", 1024)
        self.temperature = env_float("GROQ_TEMPERATURE", 0.7)
        self.history_turns = max(1, env_int("HISTORY_TURNS", 8))
        self.history: dict[int, list[dict[str, str]]] = defaultdict(list)
        self.locks: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)
        self.owner_ids: set[int] = set()

    @property
    def groq(self) -> Groq:
        if self._groq is None:
            self._groq = Groq(api_key=env_required("GROQ_API_KEY"))
        return self._groq

    async def setup_hook(self) -> None:
        guild_id = guild_id_from_env()
        if guild_id is not None:
            guild = discord.Object(id=guild_id)
            self.tree.copy_global_to(guild=guild)
            synced = await self.tree.sync(guild=guild)
            print(f"Synced {len(synced)} commands to guild {guild_id}.")
            return
        synced = await self.tree.sync()
        print(f"Synced {len(synced)} global commands.")

    async def on_ready(self) -> None:
        info = await self.application_info()
        if info.team is not None:
            self.owner_ids = {member.id for member in info.team.members}
        else:
            self.owner_ids = {info.owner.id}
        search = "on" if self.uses_web_search() else "off"
        print(f"Logged in as {self.user} (model: {self.model}, web search: {search}).")

    def uses_web_search(self) -> bool:
        return self.web_search and self.model in SEARCH_MODELS

    def can_change_model(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id in self.owner_ids:
            return True
        return interaction.guild is not None and interaction.permissions.manage_guild

    def complete(self, channel_id: int, author: str, question: str) -> str:
        history = self.history[channel_id]
        history.append({"role": "user", "content": f"{author}: {question}"})
        max_messages = self.history_turns * 2
        if len(history) > max_messages:
            del history[:-max_messages]

        request: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                *history,
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.uses_web_search():
            request["tools"] = [{"type": "browser_search"}]
            request["tool_choice"] = "auto"
            request["reasoning_effort"] = "low"
        try:
            completion = self.groq.chat.completions.create(**request)
            reply = (completion.choices[0].message.content or "").strip()
        except Exception:
            history.pop()
            raise

        if not reply:
            history.pop()
            return "(The model returned an empty reply.)"
        history.append({"role": "assistant", "content": reply})
        if len(history) > max_messages:
            del history[:-max_messages]
        return reply

    async def answer(self, channel_id: int, author: str, question: str) -> str:
        async with self.locks[channel_id]:
            return await asyncio.to_thread(self.complete, channel_id, author, question)

    def list_models(self) -> list[str]:
        page = self.groq.models.list()
        ids = sorted(model.id for model in page.data if is_chat_model(model.id))
        return ids


bot = GroqBot()


def format_error(exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    if len(message) > 500:
        message = message[:500] + "…"
    return f"Groq request failed: {message}"


async def send_chunks(send, text: str) -> None:
    for chunk in chunk_text(text):
        await send(chunk)


@bot.tree.command(name="ask", description="Ask Groq anything")
@app_commands.describe(question="Your question")
async def ask(interaction: discord.Interaction, question: str) -> None:
    question = question.strip()
    if not question:
        await interaction.response.send_message("Ask a question after the command.", ephemeral=True)
        return
    await interaction.response.defer()
    author = interaction.user.display_name
    channel_id = interaction.channel_id or interaction.user.id
    try:
        reply = await bot.answer(channel_id, author, question)
    except APIError as exc:
        await interaction.followup.send(format_error(exc))
        return
    await send_chunks(interaction.followup.send, reply)


@bot.tree.command(name="reset", description="Clear this channel's conversation memory")
async def reset(interaction: discord.Interaction) -> None:
    channel_id = interaction.channel_id or interaction.user.id
    bot.history.pop(channel_id, None)
    await interaction.response.send_message("Cleared the conversation memory for this channel.")


@bot.tree.command(name="model", description="Show or change the Groq model")
@app_commands.describe(model_id="Model id, such as openai/gpt-oss-120b")
async def model_command(
    interaction: discord.Interaction,
    model_id: str | None = None,
) -> None:
    if model_id is None:
        await interaction.response.send_message(f"Current model: `{bot.model}`")
        return
    if not bot.can_change_model(interaction):
        await interaction.response.send_message(
            "Only the bot owner, or someone with Manage Server, can change the model.",
            ephemeral=True,
        )
        return
    chosen = model_id.strip()
    if not chosen or not is_chat_model(chosen):
        await interaction.response.send_message(
            "That does not look like a Groq chat model. Try `/models`.",
            ephemeral=True,
        )
        return
    bot.model = chosen
    if not bot.web_search:
        note = "Web search is turned off."
    elif bot.uses_web_search():
        note = "This model can search the live web."
    else:
        note = "This model cannot search the web. Use `openai/gpt-oss-120b` for live lookup."
    await interaction.response.send_message(f"Now using `{bot.model}`. {note}")


@bot.tree.command(name="models", description="List Groq chat models on your account")
async def models_command(interaction: discord.Interaction) -> None:
    await interaction.response.defer(ephemeral=True)
    try:
        ids = await asyncio.to_thread(bot.list_models)
    except APIError as exc:
        await interaction.followup.send(format_error(exc), ephemeral=True)
        return
    if not ids:
        await interaction.followup.send("No chat models were returned.", ephemeral=True)
        return
    listing = "\n".join(ids)
    await interaction.followup.send(
        f"Available chat models:\n```\n{listing[:1800]}\n```",
        ephemeral=True,
    )


@bot.tree.error
async def on_tree_error(interaction: discord.Interaction, error: app_commands.AppCommandError) -> None:
    cause = error.original if isinstance(error, app_commands.CommandInvokeError) else error
    text = format_error(cause)
    if interaction.response.is_done():
        await interaction.followup.send(text)
        return
    await interaction.response.send_message(text, ephemeral=True)


@bot.event
async def on_message(message: discord.Message) -> None:
    if message.author.bot or bot.user is None:
        return

    is_dm = isinstance(message.channel, discord.DMChannel)
    mentioned = bot.user in message.mentions
    if not is_dm and not mentioned:
        return

    question = re.sub(rf"<@!?{bot.user.id}>", "", message.content).strip()
    if not question:
        prompt = "Send me a question." if is_dm else "Mention me with a question."
        await message.reply(prompt, mention_author=False)
        return

    try:
        async with message.channel.typing():
            reply = await bot.answer(message.channel.id, message.author.display_name, question)
    except APIError as exc:
        await message.reply(format_error(exc), mention_author=False)
        return

    chunks = chunk_text(reply)
    await message.reply(chunks[0], mention_author=False)
    for chunk in chunks[1:]:
        await message.channel.send(chunk)


def main() -> None:
    token = env_required("DISCORD_TOKEN")
    env_required("GROQ_API_KEY")
    guild_id_from_env()
    bot.run(token)


if __name__ == "__main__":
    main()
