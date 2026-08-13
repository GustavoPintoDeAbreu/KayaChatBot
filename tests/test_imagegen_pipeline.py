"""The two-stage edit: commit to the scene, then put the face back.

The bake-off showed both editors trading identity against instruction-following
because one prompt was doing two jobs — Kontext scored 0.918 likeness on a
prison-yard edit by declining to add the injuries it was asked for. Splitting the
work removes the trade, but only if two rules hold, and both are tested here:

* an edit that is MEANT to change the face must not have it restored, and
* a restore that does not measurably help must be thrown away.
"""
import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chat import imagegen


class FakeBackend:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def generate(self, messages, **kwargs):
        self.calls.append(messages)
        return self.reply


def _config(**over):
    icfg = {"enabled": True, "translate_prompt": True}
    icfg.update(over)
    return {"chat": {"imagegen": icfg}}


class TestFaceDecision:
    """`FACE: keep` / `FACE: change` decides whether the restore runs at all."""

    def test_keep_asks_for_a_restore(self):
        backend = FakeBackend("Dress the person as a king.\nFACE: keep")
        instruction, restore, _heavy = imagegen.build_edit_instruction(
            _config(), "põe-me de rei", backend)
        assert instruction == "Dress the person as a king."
        assert restore is True

    def test_change_skips_the_restore(self):
        backend = FakeBackend("Turn the person into a rotting zombie.\nFACE: change")
        instruction, restore, _heavy = imagegen.build_edit_instruction(
            _config(), "faz dele um zombie", backend)
        assert instruction == "Turn the person into a rotting zombie."
        assert restore is False

    def test_the_face_line_never_reaches_the_image_model(self):
        backend = FakeBackend("Put a crown on the person.\nFACE: keep")
        instruction, _, _heavy = imagegen.build_edit_instruction(
            _config(), "põe-lhe uma coroa", backend)
        assert "FACE" not in instruction

    @pytest.mark.parametrize("reply", [
        "Dress the person as a king.",              # no FACE line at all
        "Dress the person as a king.\nFACE: maybe",  # unparseable value
        "Dress the person as a king.\nFACE:",        # truncated
    ])
    def test_a_malformed_answer_skips_the_restore(self, reply):
        """A stage that changes the output must not fire on a reply we could not read."""
        instruction, restore, _heavy = imagegen.build_edit_instruction(
            _config(), "põe-me de rei", FakeBackend(reply))
        assert instruction == "Dress the person as a king."
        assert restore is False

    def test_case_and_spacing_are_tolerated(self):
        backend = FakeBackend("Put him on a beach.\n  face :  KEEP  ")
        _, restore, _heavy = imagegen.build_edit_instruction(_config(), "praia", backend)
        assert restore is True

    def test_translation_disabled_returns_the_raw_text_and_no_restore(self):
        instruction, restore, _heavy = imagegen.build_edit_instruction(
            _config(translate_prompt=False), "põe-me de rei", FakeBackend("x"))
        assert instruction == "põe-me de rei"
        assert restore is False

    def test_a_backend_failure_falls_back_without_restoring(self):
        class Boom:
            def generate(self, *a, **kw):
                raise RuntimeError("model down")

        instruction, restore, _heavy = imagegen.build_edit_instruction(
            _config(), "põe-me de rei", Boom())
        assert instruction == "põe-me de rei"
        assert restore is False


class TestJpegDelivery:
    """PNG is re-encoded before it is base64-inlined into the WAHA JSON body."""

    @staticmethod
    def _png(size=(64, 64)):
        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGB", size, (120, 30, 30)).save(buffer, format="PNG")
        return buffer.getvalue()

    def test_png_becomes_jpeg(self):
        out = imagegen._as_jpeg(self._png())
        assert out[:2] == b"\xff\xd8", "expected a JPEG SOI marker"

    def test_it_gets_smaller(self):
        from PIL import Image

        buffer = io.BytesIO()
        # Noise, so PNG cannot win on flat colour.
        import random

        random.seed(0)
        image = Image.new("RGB", (400, 400))
        image.putdata([(random.randint(0, 255),) * 3 for _ in range(400 * 400)])
        image.save(buffer, format="PNG")
        png = buffer.getvalue()
        assert len(imagegen._as_jpeg(png)) < len(png)

    def test_garbage_is_returned_untouched(self):
        """A heavier image is better than no image."""
        assert imagegen._as_jpeg(b"not an image") == b"not an image"

    def test_send_image_uses_the_format_it_is_given(self):
        from src.chat.waha_client import MockWahaClient

        client = MockWahaClient(echo=False)
        # The mock records the byte count; what matters is that the kwargs exist
        # and are accepted, since the real client puts them in the JSON body.
        client.send_image("chat", b"xx", mimetype="image/jpeg", filename="kaya.jpg")
        assert client.sent[-1]["image_bytes"] == 2

    def test_the_real_client_sends_the_declared_mimetype(self):
        import base64

        from src.chat.waha_client import WahaClient

        posted = {}

        class FakeHttp:
            def post(self, url, json=None, timeout=None):
                posted["url"] = url
                posted["body"] = json

                class R:
                    @staticmethod
                    def raise_for_status():
                        return None

                    @staticmethod
                    def json():
                        return {"id": "1"}

                return R()

        client = WahaClient.__new__(WahaClient)
        client.session = "default"
        client._client = FakeHttp()
        client.send_image("chat", b"abc", mimetype="image/jpeg", filename="kaya.jpg")

        assert posted["body"]["file"]["mimetype"] == "image/jpeg"
        assert posted["body"]["file"]["filename"] == "kaya.jpg"
        assert base64.b64decode(posted["body"]["file"]["data"]) == b"abc"


class TestRestoreGuard:
    """The restore is kept only when it beats the editor's own output."""

    @staticmethod
    def _worker():
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "imagegen_worker",
            Path(__file__).parent.parent / "scripts" / "imagegen_worker.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_a_worse_restore_is_discarded(self, monkeypatch):
        worker = self._worker()
        from PIL import Image

        edited = Image.new("RGB", (32, 32), (10, 10, 10))
        restored = Image.new("RGB", (32, 32), (250, 250, 250))

        scores = {id(edited): 0.60, id(restored): 0.40}
        monkeypatch.setattr(worker, "open_source_photo", lambda path: edited)

        import src.chat.face_utils as face_utils

        monkeypatch.setattr(face_utils, "likeness",
                            lambda ref, img: scores.get(id(img)))
        monkeypatch.setattr(worker, "_run_bfs", lambda *a, **kw: restored)

        out, info = worker.restore_with_bfs(edited, "src.jpg", object(), {})
        assert out is edited, "a restore that scored lower must be thrown away"
        assert info["restored"] is False

    def test_a_better_restore_is_kept(self, monkeypatch):
        worker = self._worker()
        from PIL import Image

        edited = Image.new("RGB", (32, 32), (10, 10, 10))
        restored = Image.new("RGB", (32, 32), (250, 250, 250))
        scores = {id(edited): 0.30, id(restored): 0.72}

        monkeypatch.setattr(worker, "open_source_photo", lambda path: edited)

        import src.chat.face_utils as face_utils

        monkeypatch.setattr(face_utils, "likeness",
                            lambda ref, img: scores.get(id(img)))
        monkeypatch.setattr(worker, "_run_bfs", lambda *a, **kw: restored)

        out, info = worker.restore_with_bfs(edited, "src.jpg", object(), {})
        assert out is restored
        assert info["restored"] is True
        assert info["likeness"] == 0.72

    def test_an_unmeasurable_restore_keeps_the_edit(self, monkeypatch):
        """No face found in one of them — do not gamble on an unscored swap."""
        worker = self._worker()
        from PIL import Image

        edited = Image.new("RGB", (32, 32), (10, 10, 10))
        monkeypatch.setattr(worker, "open_source_photo", lambda path: edited)

        import src.chat.face_utils as face_utils

        monkeypatch.setattr(face_utils, "likeness", lambda ref, img: None)
        monkeypatch.setattr(worker, "_run_bfs", lambda *a, **kw: edited)

        out, info = worker.restore_with_bfs(edited, "src.jpg", object(), {})
        assert out is edited
        assert info["restored"] is False

    def test_a_crashing_restore_keeps_the_edit(self, monkeypatch):
        worker = self._worker()
        from PIL import Image

        edited = Image.new("RGB", (32, 32), (10, 10, 10))

        def boom(*a, **kw):
            raise RuntimeError("no VRAM")

        monkeypatch.setattr(worker, "_run_bfs", boom)
        out, info = worker.restore_with_bfs(edited, "src.jpg", object(), {})
        assert out is edited
        assert info["restored"] is False
        assert "restore_error" in info


class TestEditStrength:
    """A costume swap and "make these two kiss" were asked at the same guidance,
    and the second came back as the original photo."""

    def test_a_pose_change_is_heavy(self):
        backend = FakeBackend("Make the two men kiss each other.\n"
                              "FACE: keep\nEDIT: heavy")
        instruction, restore, heavy = imagegen.build_edit_instruction(
            _config(), "põe estes dois a beijarem-se", backend)
        assert instruction == "Make the two men kiss each other."
        assert restore is True and heavy is True

    def test_a_costume_swap_is_light(self):
        backend = FakeBackend("Dress the person as a king.\nFACE: keep\nEDIT: light")
        _, _, heavy = imagegen.build_edit_instruction(_config(), "põe-me de rei", backend)
        assert heavy is False

    def test_the_edit_line_never_reaches_the_image_model(self):
        backend = FakeBackend("Put a gun in his hand.\nFACE: keep\nEDIT: heavy")
        instruction, _, _ = imagegen.build_edit_instruction(
            _config(), "põe-lhe uma arma na mão", backend)
        assert "EDIT" not in instruction and "FACE" not in instruction

    @pytest.mark.parametrize("reply", [
        "Dress the person as a king.\nFACE: keep",              # no EDIT line
        "Dress the person as a king.\nFACE: keep\nEDIT: maybe",  # unparseable
        "Dress the person as a king.\nFACE: keep\nEDIT:",        # truncated
    ])
    def test_a_malformed_answer_keeps_the_ordinary_guidance(self, reply):
        """A setting that changes the output must not fire on an unreadable reply."""
        _, _, heavy = imagegen.build_edit_instruction(
            _config(), "põe-me de rei", FakeBackend(reply))
        assert heavy is False

    def test_the_lines_may_arrive_in_either_order(self):
        backend = FakeBackend("Make them kiss.\nEDIT: heavy\nFACE: keep")
        instruction, restore, heavy = imagegen.build_edit_instruction(
            _config(), "beijo", backend)
        assert instruction == "Make them kiss." and restore is True and heavy is True

    def test_a_backend_failure_is_never_heavy(self):
        class Boom:
            def generate(self, *a, **kw):
                raise RuntimeError("model down")

        _, _, heavy = imagegen.build_edit_instruction(_config(), "beijo", Boom())
        assert heavy is False
