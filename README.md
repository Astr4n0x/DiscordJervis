<p align="center">
  <a href="https://discord.com/developers/applications"><img alt="Discord" src="https://img.shields.io/badge/Discord-bot-5865F2?style=flat-square&logo=discord&logoColor=white"></a>
  <a href="https://www.python.org/"><img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white"></a>
  <a href="https://console.groq.com/"><img alt="Groq" src="https://img.shields.io/badge/Groq-gpt--oss--120b-F55036?style=flat-square"></a>
</p>

<h1 align="center">discord-groq-bot</h1>

<p align="center">
  A small Discord bot that answers with Groq.<br>
  <code>/ask</code> in a channel, mention it, or just DM it.
</p>

---

## Talk to it

| Where | How |
| --- | --- |
| Any channel | `/ask` and the question |
| Any channel | `@the-bot` plus the question |
| Its DMs | Type the question. No mention. |

It keeps the last few turns for that channel, so a follow-up can refer to the message before it. `/reset` wipes that channel’s memory. Each channel has its own.

The stock model is `openai/gpt-oss-120b`. Questions about news, prices, weather, scores, and releases can be looked up on the web first. The link it used shows up in the reply. Everyday questions, including code, are answered straight away. `/model` swaps the model until you stop the process. Set `GROQ_MODEL` if you want the swap to survive a restart. `openai/gpt-oss-20b` can search too. Other Groq models can’t.

Long replies are cut into Discord-sized messages. Headings, lists, quotes, code, and links are written the way Discord actually renders them.

## Get it running

You need Python 3.10 or newer, a Discord application, and a Groq key.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

On macOS or Linux, activate with `source .venv/bin/activate` and copy the example with `cp .env.example .env`.

Fill in `.env`:

| Name | What to put there |
| --- | --- |
| `DISCORD_TOKEN` | Developer Portal → your app → **Bot** → token |
| `GROQ_API_KEY` | [console.groq.com/keys](https://console.groq.com/keys) |
| `DISCORD_GUILD_ID` | Optional. Your server id, if you want slash commands in that server immediately |

Invite the bot. Replace `CLIENT_ID` with the application id from the portal.

```
https://discord.com/oauth2/authorize?client_id=CLIENT_ID&scope=bot+applications.commands&permissions=84992
```

Those permissions are View Channels, Send Messages, Embed Links, and Read Message History. Mentions and DMs work without the Message Content intent.

```powershell
python bot.py
```

You should see it log in, with the model name and `web search: on`. Global slash commands can take a while to show up. A guild id skips that wait for one server.

Run a single copy. A second process with the same token kicks the first one off Discord.

## Commands

| Command | Who | Effect |
| --- | --- | --- |
| `/ask` | anyone | Ask a question |
| `/reset` | anyone | Forget this channel |
| `/model` | anyone | Show the model in use |
| `/model <id>` | bot owner, or Manage Server | Use a different Groq model until restart |
| `/models` | anyone | List chat models on your Groq account. Only you see the list |

## Knobs

All of these live in `.env`. Skip a line and the default is used.

| Name | Default | What it changes |
| --- | --- | --- |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Model used at startup |
| `WEB_SEARCH` | `true` | Live lookup on the GPT-OSS models |
| `GROQ_MAX_TOKENS` | `1024` | Length cap on a reply |
| `GROQ_TEMPERATURE` | `0.7` | How loose the wording is |
| `HISTORY_TURNS` | `8` | How many exchanges a channel remembers |
| `SYSTEM_PROMPT` | see `.env.example` | How the bot is told to talk |
| `DISCORD_GUILD_ID` | empty | Sync slash commands to one server |

`WEB_SEARCH=false` turns lookup off. The bot still answers.

## Files

```
bot.py            the bot
requirements.txt  discord.py, groq, python-dotenv
.env.example      names of every setting
```

`.env` stays on your machine. It is listed in `.gitignore`.
