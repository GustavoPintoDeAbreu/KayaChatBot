"""The describer saying it cannot see anything must not become group memory.

A refusal comes back as a successful HTTP 200, so nothing treated it as a
failure: the live log holds "[Imagem: Por favor, fornece o vídeo ou a imagem
para que eu possa descrevê-la]" twice, recorded as something a member said and
ready to be ingested into the vector store.
"""
import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat.vision import flatten_animation, looks_like_a_refusal
from src.data.ingest import strip_failed_descriptions

# The two that actually reached production, plus the shapes around them.
REFUSALS = [
    "Por favor, fornece o vídeo ou a imagem para que eu possa descrevê-la.",
    "Como não forneceste o vídeo ou a imagem, não consigo descrevê-la. "
    "Por favor, partilha o ficheiro para que eu possa realizar a tarefa.",
    "Não consigo ver a imagem.",
    "No image was provided.",
    "I cannot see any image here.",
    "",
    "   ",
]

DESCRIPTIONS = [
    "A imagem mostra um eclipse solar total num céu escuro e granulado, onde a "
    "lua cobre completamente o sol.",
    "Três pessoas posam para uma fotografia em frente a uma parede de pedra clara, "
    "num ambiente soalheiro.",
    "A imagem apresenta uma tabela informativa sobre um fundo preto, sem pessoas visíveis.",
]


@pytest.mark.parametrize("text", REFUSALS)
def test_a_refusal_is_recognised(text):
    assert looks_like_a_refusal(text) is True


@pytest.mark.parametrize("text", DESCRIPTIONS)
def test_a_real_description_is_kept(text):
    assert looks_like_a_refusal(text) is False


def test_a_failed_description_is_dropped_before_ingestion():
    poisoned = "[Imagem: Por favor, fornece o vídeo ou a imagem para que eu possa descrevê-la.]"
    assert strip_failed_descriptions(poisoned) == ""


def test_a_caption_survives_its_failed_description():
    text = "olha esta foto\n[Imagem: Como não forneceste a imagem, não consigo descrevê-la.]"
    assert strip_failed_descriptions(text) == "olha esta foto"


def test_a_real_description_survives_ingestion():
    text = "[Imagem: A imagem mostra um eclipse solar total num céu escuro e granulado.]"
    assert strip_failed_descriptions(text) == text


def test_text_without_an_image_is_untouched():
    assert strip_failed_descriptions("texto normal") == "texto normal"


# ── animated stickers ────────────────────────────────────────────────────────
def _webp(frames: int) -> bytes:
    from PIL import Image

    images = [Image.new("RGB", (32, 32), (index * 40, 0, 0)) for index in range(frames)]
    buffer = io.BytesIO()
    images[0].save(buffer, format="WEBP", save_all=frames > 1,
                   append_images=images[1:], duration=100)
    return buffer.getvalue()


def test_an_animated_sticker_is_flattened_to_one_frame():
    """The projector cannot decode animation, which is what produced the two
    poisoned lines in the first place."""
    data, mimetype = flatten_animation(_webp(3), "image/webp")

    assert mimetype == "image/png"
    from PIL import Image

    with Image.open(io.BytesIO(data)) as flat:
        assert getattr(flat, "is_animated", False) is False


def test_a_static_sticker_passes_through_untouched():
    original = _webp(1)
    assert flatten_animation(original, "image/webp") == (original, "image/webp")


def test_a_photo_is_never_re_encoded():
    assert flatten_animation(b"not-an-image", "image/jpeg") == (b"not-an-image", "image/jpeg")


def test_undecodable_bytes_are_left_alone():
    """A failed describe is better than a dropped message."""
    assert flatten_animation(b"garbage", "image/webp") == (b"garbage", "image/webp")
