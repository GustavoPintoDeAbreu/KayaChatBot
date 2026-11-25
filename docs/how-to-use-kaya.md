# How to use Kaya in the group

Kaya is the group's assistant: it remembers the group's history and answers in
European Portuguese or English. This is the owner's reference for everything it
can do, so the commands are not lost as the lab grows. The same list, shorter,
comes back on WhatsApp with `/help`, in the group too: each chat gets only what
works there (the group never sees the owner's commands).

Since 2026-11-25 that list is not hand-written: every repo declares its commands
in its `lab.yaml` (`commands:`, with `where: [group, dm]` and `who: everyone|owner`),
labwatch collects them (`/api/commands`, the Help page), and a test in each repo
fails if its parser and its `lab.yaml` disagree. A new command is one line there.

## In the group

Everything below works in the group and in a DM with Kaya's number (unless it
says otherwise).

### Slash commands (exact)

- `/bug <texto>` (`/erro`) — report a problem (the reply is fixed, nothing generated).
- `/feedback <texto>` (`/sugestao`, `/sugestão`) — a suggestion.
- `/clear` (`/limpar`) — forget this chat's recent context so the next message
  starts fresh.
- `/aniversario <dia/mês>` (`/birthday`, `/anos`) — state your own birthday;
  the most trusted source there is.
- `/help` (`/ajuda`) — the commands this chat can use.

These are matched literally and never reach the model; a bare `/bug` or
`/feedback` explains itself and stores nothing. A `/word` nobody handles gets
*"Não conheço /word. Querias dizer /X? Manda /help…"*, the guess taken only from
what that chat can use.

### Voice and text

Kaya answers in voice or text, per chat:

- "responde só em áudio" / "fala comigo por voz" — switch this chat to voice.
- "volta a responder por texto" / "chega de áudios, escreve" — switch back.
- "responde a esta por voz" / "explica isso num áudio" — just this one answer
  as a voice note, without changing the default.

### Recaps

- "o que é que decidimos sobre o almoço de natal?" — sum up a topic: what was
  decided, what is open, who wanted what.
- "o que é que decidimos?" — recap this chat's recent discussion.

### Polls

- "faz uma sondagem com as opções" / "cria uma votação" — make a native
  WhatsApp poll from what the chat has been discussing. The poll itself is a
  real WhatsApp poll, never generated text.
- "quem está a ganhar a votação?" / "quem falta votar?" — the result, counted
  by code from the recorded votes.

### Counting

- "quantas vezes é que o X disse Y?" / "quem diz mais palavrões?" — a tally of
  messages, counted by code (top-k memory cannot count).

### Documents

Share a PDF in the chat and Kaya reads it (the synopsis is written by the local
model; nothing leaves the box). Ask about it later:

- "que documento era aquele?" / "qual era o PDF sobre X?" — finds it and cites
  it by page.

### Web search and facts

- A question about the world ("quem é melhor, Ronaldo ou Messi?") is answered
  without the group's data, and may use the web. A query that names a group
  member never leaves the box.
- A request to judge an argument ("quem tem razão sobre X?") routes to debate:
  it cites what it was handed and nothing it invented.

### Images

Photos sent to the chat are described and become searchable memory ("aquela
foto do barco"). Making or editing images was removed; a request to make one
gets a fixed refusal, not a promise.

## In your DM (owner only)

These are handled by the Pi gateway, so they work even while the PC is off:

- `/help` — everything, your commands included, with a link to the Help page.
  A wrong `/word` here is answered by the Pi with the closest command.
- `/idea <texto>` — send an idea to the idea pipeline (turns it into a planned,
  implemented draft PR); flags `!new !local !uncensored !claude !deepseek !glm`.
- `/idea list` · `/idea pending` · `/idea merge all` · `/idea help` ·
  `/idea allow <site>` — look around, merge, help, allow a site.
- `/homelaboff` — ask what is running on the PC and shut it down once free.
- `/homelaboff_bruteforce` — shut the PC down at once, skipping every wait (asks for a yes).
- `/homelabon` — wake the PC (Wake-on-LAN) and send its Claude session link.
- `/gamingoff` — turn the Windows gaming PC off (asks for a yes).
- `/homelabrc [home]` / `/homelabrc stop [home]` — start/end a Claude Remote
  Control session on the PC.
- `/homelabqc [home]` / `/homelabqc stop [home]` — start/end a Qwen Code
  (qcode) Web Shell session on the PC.

Anything else in your DM — a voice note or plain text — goes to the lab
assistant, which answers lab questions and hands group questions back to Kaya.

## The Help page

The labwatch Help page lists every project and what it can do, with this doc
rendered in full and a WhatsApp filter. It is the single place to look when a
command is forgotten.
