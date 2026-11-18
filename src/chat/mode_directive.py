"""A reply-mode instruction at the END of a message: "…, answer in audio", "responde em texto".

Gustavo's way of talking to the bot (2026-11-18): any message, typed or spoken, may end with
how he wants the answer. "Answer in audio" switches the bot to voice notes until "answer in
text"; a message that is only the instruction gets a short confirmation and nothing else (it
must not re-answer the conversation). Found here in code, before the router, because the router
gives one label per message: "who's coming tonight? answer in audio" came back either as a
question (no switch) or as CMD_AUDIO (the question dropped).

Strict on purpose: the instruction must be the message's last clause, an imperative to the bot
("answer / reply / responde / fala …") naming a medium. Talking *about* audio ("o Rafa respondeu
em áudio", "what did he say in the audio") is not an instruction. Anything else is left to the
router's CMD_AUDIO / CMD_TEXT / CMD_AUDIO_ONCE, as before.

Stdlib only: the Pi gateway imports the chat package's light modules.
"""
from __future__ import annotations

import re
import unicodedata
from typing import NamedTuple, Optional

AUDIO, TEXT, AUDIO_ONCE = "audio", "text", "audio_once"

_VERB = (r"(?:answer|reply|respond|responde|responda|respondes|responder|continua a responder|volta a responder)")
_VERB_ME = (r"(?:talk|speak|write|send it|say it|fala|falas|falar|escreve|escrevas|escrever|"
            r"manda|envia|diz|continua|text|message)")
_ME = r"(?:\s*-?\s*(?:me|to me|comigo|nos|connosco))"
_CONN = r"(?:in|by|via|with|as|over|em|por|com|num|numa|no|na|pelo|pela|atraves de)"
_ART = r"(?:\s+(?:a|an|um|uma|o|a))?"
_AUDIO = r"(?:audio|audios|voice notes?|voice messages?|voice|voz|notas? de voz|mensagens? de voz)"
_TEXT = r"(?:text|texts|texto|textos|writing|escrito|mensagem escrita|mensagens escritas)"
_TAIL = (r"(?:\s*,?\s*(?:from now on|for now|again|only|please|pls|plz|"
         r"a partir de agora|daqui para a frente|daqui em diante|outra vez|de novo|por favor|pf|sff|so|apenas))*")
_ONCE = r"(?:just|only|so|apenas)?\s*(?:this one|this time|this answer|this message|esta|esta vez|desta vez|so esta|esta resposta)"

_STANDING = re.compile(
    rf"(?:^|[\s,.;:!?()-])(?P<cmd>"
    rf"(?:(?:please|por favor|pf|sff|from now on|a partir de agora|daqui para a frente|daqui em diante)\s*,?\s*)*"
    rf"(?:(?:{_VERB}{_ME}?|{_VERB_ME}{_ME})\s+{_CONN}{_ART}\s+(?P<m1>{_AUDIO}|{_TEXT})"
    rf"|(?:back to|go back to|switch to|change to|volta ao|volta para o|volta pro|volta a|passa a|passa para|muda para)\s+(?:the\s+|o\s+)?(?P<m2>{_AUDIO}|{_TEXT}|escrever|responder por escrito)"
    rf"|(?:only|just|so|apenas)\s+(?P<m3>{_AUDIO}|{_TEXT})"
    rf"|(?P<m4>{_AUDIO}|{_TEXT})\s+(?:only|please|por favor|so|apenas))"
    rf"{_TAIL}(?P<once>\s*,?\s*{_ONCE})?)\s*[.!?…]*\s*$")


_CORE = re.compile(
    rf"(?:(?:{_VERB}{_ME}?|{_VERB_ME}{_ME})\s+{_CONN}{_ART}\s+(?:{_AUDIO}|{_TEXT})"
    rf"|(?:back to|go back to|switch to|volta ao|volta a|passa a)\s+(?:the\s+|o\s+)?(?:{_AUDIO}|{_TEXT}|escrever))")
_FILLER = re.compile(
    r"\b(?:ok(?:ay)?|okey|thanks|thank you|thx|cheers|please|pls|just this message|only this message|"
    r"this message|nothing else|that'?s it|that'?s all|from now on|again|"
    r"obrigad[oa]|valeu|pronto|so isso|apenas isso|e mais nada|mais nada|so esta mensagem|esta mensagem|"
    r"a partir de agora|por favor|outra vez|de novo)\b")


class Directive(NamedTuple):
    rest: str               # the message without the instruction ("" when it was only that)
    mode: str               # AUDIO, TEXT or AUDIO_ONCE


def _fold(text: str) -> str:
    """Lowercase, accents off, so "áudio" and "audio" read the same (positions are kept)."""
    out = []
    for ch in text:
        base = unicodedata.normalize("NFKD", ch)
        out.append("".join(c for c in base if not unicodedata.combining(c))[:1] or ch)
    return "".join(out).lower()


def find(text: str) -> Optional[Directive]:
    """The trailing reply-mode instruction of ``text``, if it ends with one."""
    if not text or not text.strip():
        return None
    folded = _fold(text)
    if len(folded) != len(text):          # only fold 1:1, so the cut lands on the original
        return None
    m = _STANDING.search(folded)
    if not m:
        return None
    loose = bool(m.group("m3") or m.group("m4"))
    medium = next(m.group(g) for g in ("m1", "m2", "m3", "m4") if m.group(g))
    mode = TEXT if re.match(_TEXT + r"|escrever|responder por escrito", medium) else AUDIO
    if m.group("once") and mode == AUDIO:
        mode = AUDIO_ONCE
    rest = text[:m.start("cmd")].rstrip(" ,;:-–—").strip()
    if loose and rest:
        return None
    # "Okay thanks. Answer in text, just this message, nothing else. Answer in text." is still
    # only the instruction: what's left is fillers and the instruction said again.
    residue = _FILLER.sub(" ", _CORE.sub(" ", _fold(rest)))
    if not re.search(r"[a-z0-9]", residue):
        rest = ""
    return Directive(rest, mode)


_ENGLISH = re.compile(r"\b(?:answer|reply|respond|talk|speak|write|send|voice|text only|back to|switch to|"
                      r"from now on|please|just this|only)\b")
_PORTUGUESE = re.compile(r"\b(?:responde|responda|fala|escreve|manda|envia|volta|texto|voz|comigo|"
                         r"a partir de agora|por favor|so esta|desta vez)\b")


def language(text: str) -> str:
    """"en" or "pt" for the confirmation, from the instruction's own words (short messages
    like "answer in text" carry too little for a general language guess)."""
    folded = _fold(text or "")
    en, pt = len(_ENGLISH.findall(folded)), len(_PORTUGUESE.findall(folded))
    return "en" if en > pt else "pt"
