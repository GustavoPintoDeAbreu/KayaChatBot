"""The two-stage edit: commit to the scene, then put the face back.

The bake-off showed both editors trading identity against instruction-following
because one prompt was doing two jobs — Kontext scored 0.918 likeness on a
prison-yard edit by declining to add the injuries it was asked for. Splitting the
work removes the trade, but only if two rules hold, and both are tested here:

* an edit that is MEANT to change the face must not have it restored, and
* a restore that does not measurably help must be thrown away.
"""
import io
import json
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
        instruction, restore, _heavy, _subject = imagegen.build_edit_instruction(
            _config(), "põe-me de rei", backend)
        assert instruction == "Dress the person as a king."
        assert restore is True

    def test_change_skips_the_restore(self):
        backend = FakeBackend("Turn the person into a rotting zombie.\nFACE: change")
        instruction, restore, _heavy, _subject = imagegen.build_edit_instruction(
            _config(), "faz dele um zombie", backend)
        assert instruction == "Turn the person into a rotting zombie."
        assert restore is False

    def test_the_face_line_never_reaches_the_image_model(self):
        backend = FakeBackend("Put a crown on the person.\nFACE: keep")
        instruction, _, _heavy, _subject = imagegen.build_edit_instruction(
            _config(), "põe-lhe uma coroa", backend)
        assert "FACE" not in instruction

    @pytest.mark.parametrize("reply", [
        "Dress the person as a king.",              # no FACE line at all
        "Dress the person as a king.\nFACE: maybe",  # unparseable value
        "Dress the person as a king.\nFACE:",        # truncated
    ])
    def test_a_malformed_answer_skips_the_restore(self, reply):
        """A stage that changes the output must not fire on a reply we could not read."""
        instruction, restore, _heavy, _subject = imagegen.build_edit_instruction(
            _config(), "põe-me de rei", FakeBackend(reply))
        assert instruction == "Dress the person as a king."
        assert restore is False

    def test_case_and_spacing_are_tolerated(self):
        backend = FakeBackend("Put him on a beach.\n  face :  KEEP  ")
        _, restore, _heavy, _subject = imagegen.build_edit_instruction(_config(), "praia", backend)
        assert restore is True

    def test_translation_disabled_returns_the_raw_text_and_no_restore(self):
        instruction, restore, _heavy, _subject = imagegen.build_edit_instruction(
            _config(translate_prompt=False), "põe-me de rei", FakeBackend("x"))
        assert instruction == "põe-me de rei"
        assert restore is False

    def test_a_backend_failure_falls_back_without_restoring(self):
        class Boom:
            def generate(self, *a, **kw):
                raise RuntimeError("model down")

        instruction, restore, _heavy, _subject = imagegen.build_edit_instruction(
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
        instruction, restore, heavy, _subject = imagegen.build_edit_instruction(
            _config(), "põe estes dois a beijarem-se", backend)
        assert instruction == "Make the two men kiss each other."
        assert restore is True and heavy is True

    def test_a_costume_swap_is_light(self):
        backend = FakeBackend("Dress the person as a king.\nFACE: keep\nEDIT: light")
        _, _, heavy, _subject = imagegen.build_edit_instruction(_config(), "põe-me de rei", backend)
        assert heavy is False

    def test_the_edit_line_never_reaches_the_image_model(self):
        backend = FakeBackend("Put a gun in his hand.\nFACE: keep\nEDIT: heavy")
        instruction, _, _, _subject = imagegen.build_edit_instruction(
            _config(), "põe-lhe uma arma na mão", backend)
        assert "EDIT" not in instruction and "FACE" not in instruction

    @pytest.mark.parametrize("reply", [
        "Dress the person as a king.\nFACE: keep",              # no EDIT line
        "Dress the person as a king.\nFACE: keep\nEDIT: maybe",  # unparseable
        "Dress the person as a king.\nFACE: keep\nEDIT:",        # truncated
    ])
    def test_a_malformed_answer_keeps_the_ordinary_guidance(self, reply):
        """A setting that changes the output must not fire on an unreadable reply."""
        _, _, heavy, _subject = imagegen.build_edit_instruction(
            _config(), "põe-me de rei", FakeBackend(reply))
        assert heavy is False

    def test_the_lines_may_arrive_in_either_order(self):
        backend = FakeBackend("Make them kiss.\nEDIT: heavy\nFACE: keep")
        instruction, restore, heavy, _subject = imagegen.build_edit_instruction(
            _config(), "beijo", backend)
        assert instruction == "Make them kiss." and restore is True and heavy is True

    def test_a_backend_failure_is_never_heavy(self):
        class Boom:
            def generate(self, *a, **kw):
                raise RuntimeError("model down")

        _, _, heavy, _subject = imagegen.build_edit_instruction(_config(), "beijo", Boom())
        assert heavy is False


class TestSubject:
    """A photo of four Monster cans on a shop counter went through the portrait
    pipeline: rewritten as a person edit, then told to preserve a face that was
    not in the picture. The group's verdict was "ficou horrívelmente má"."""

    def test_an_object_edit_is_recognised(self):
        backend = FakeBackend("Replace the cans with dildos.\n"
                              "SUBJECT: object\nFACE: keep\nEDIT: heavy")
        instruction, _, _, subject = imagegen.build_edit_instruction(
            _config(), "transform the monster cans into dildos", backend)
        assert instruction == "Replace the cans with dildos."
        assert subject == "object"

    def test_a_scene_edit_is_recognised(self):
        backend = FakeBackend("Make it snow.\nSUBJECT: scene\nFACE: keep\nEDIT: light")
        _, _, _, subject = imagegen.build_edit_instruction(
            _config(), "mete isto a nevar", backend)
        assert subject == "scene"

    def test_the_subject_line_never_reaches_the_image_model(self):
        backend = FakeBackend("Replace the cans.\nSUBJECT: object\nFACE: keep")
        instruction, _, _, _ = imagegen.build_edit_instruction(
            _config(), "muda as latas", backend)
        assert "SUBJECT" not in instruction

    @pytest.mark.parametrize("reply", [
        "Dress the person as a king.\nFACE: keep",                 # no SUBJECT line
        "Dress the person as a king.\nSUBJECT: banana\nFACE: keep",  # unparseable
        "Dress the person as a king.\nSUBJECT:",                    # truncated
    ])
    def test_an_unreadable_subject_falls_back_to_person(self, reply):
        """`person` is what shipped, so an unparsed answer must not change behaviour."""
        _, _, _, subject = imagegen.build_edit_instruction(
            _config(), "põe-me de rei", FakeBackend(reply))
        assert subject == imagegen.SUBJECT_PERSON

    def test_a_backend_failure_falls_back_to_person(self):
        class Boom:
            def generate(self, *a, **kw):
                raise RuntimeError("model down")

        _, _, _, subject = imagegen.build_edit_instruction(_config(), "x", Boom())
        assert subject == imagegen.SUBJECT_PERSON

    def test_translation_disabled_falls_back_to_person(self):
        _, _, _, subject = imagegen.build_edit_instruction(
            _config(translate_prompt=False), "muda as latas", FakeBackend("x"))
        assert subject == imagegen.SUBJECT_PERSON


class TestEditorRouting:
    """The bake-off that chose one editor scored only faces, so it chose on
    identity preservation — a criterion an object edit does not have."""

    @staticmethod
    def _routed():
        return _config(editor="flux-kontext", editors={
            "person": "flux-kontext", "object": "flux2-klein", "scene": "flux2-klein"})

    @pytest.mark.parametrize("subject,expected", [
        ("person", "flux-kontext"),
        ("object", "flux2-klein"),
        ("scene", "flux2-klein"),
    ])
    def test_subject_picks_the_editor(self, subject, expected):
        assert imagegen.resolve_editor(self._routed(), subject) == expected

    def test_an_unmapped_subject_uses_the_fallback(self):
        assert imagegen.resolve_editor(self._routed(), "sculpture") == "flux-kontext"

    def test_no_mapping_at_all_uses_the_fallback(self):
        """Existing configs have `editor` and no `editors` — they must not change."""
        cfg = _config(editor="flux-kontext")
        assert imagegen.resolve_editor(cfg, "object") == "flux-kontext"


class TestIdentityMachineryIsGated:
    """Everything that protects a likeness is dead weight without one, and the
    identity clause is actively harmful: per config.yaml's own note it is the
    instruction a model best satisfies by changing nothing."""

    @staticmethod
    def _run(monkeypatch, tmp_path, subject, mode="edit"):
        """Capture the worker argv without launching a worker."""
        captured = {}

        class Result:
            returncode = 0
            stdout = '{"ok": true, "edited": true}'
            stderr = ""

        def fake_run(command, **kwargs):
            captured["command"] = command
            # run() checks the output file exists before reading it.
            out = Path(command[command.index("--out") + 1])
            from PIL import Image

            Image.new("RGB", (8, 8)).save(out)
            return Result()

        monkeypatch.setattr(imagegen.subprocess, "run", fake_run)
        source = tmp_path / "in.png"
        from PIL import Image

        Image.new("RGB", (8, 8)).save(source)
        imagegen.run(
            _config(enabled=True, editor="flux-kontext",
                    editors={"person": "flux-kontext", "object": "flux2-klein"},
                    keep_outputs=0),
            "an instruction", mode=mode, image_path=str(source), subject=subject)
        return captured["command"]

    def test_an_object_edit_drops_the_clause_the_crop_and_best_of_n(
            self, monkeypatch, tmp_path):
        command = self._run(monkeypatch, tmp_path, "object")
        assert "--no-identity-clause" in command
        assert "--no-face-crop" in command
        assert command[command.index("--candidates") + 1] == "1"
        assert command[command.index("--editor") + 1] == "flux2-klein"

    def test_a_person_edit_keeps_all_of_it(self, monkeypatch, tmp_path):
        command = self._run(monkeypatch, tmp_path, "person")
        assert "--no-identity-clause" not in command
        assert "--no-face-crop" not in command
        assert "--candidates" not in command
        assert command[command.index("--editor") + 1] == "flux-kontext"


class TestFaceCountDrivesTheClause:
    """The worker's own guard, independent of what the caller asked for."""

    @staticmethod
    def _worker():
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "imagegen_worker",
            Path(__file__).parent.parent / "scripts" / "imagegen_worker.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _icfg(self, worker):
        return {"identity_clause": worker.DEFAULT_IDENTITY_CLAUSE,
                "identity_clause_plural": worker.DEFAULT_IDENTITY_CLAUSE_PLURAL}

    def test_a_photo_with_nobody_in_it_gets_no_clause(self):
        """The Monster-cans bug, in one line: `0 > 1` is false, so zero faces was
        taking the same branch as one face."""
        worker = self._worker()
        assert worker.select_identity_clause(0, self._icfg(worker)) == ""

    def test_one_face_gets_the_singular_clause(self):
        worker = self._worker()
        assert "the person's face" in worker.select_identity_clause(
            1, self._icfg(worker))

    def test_two_faces_get_the_plural_clause(self):
        worker = self._worker()
        assert "every person's face" in worker.select_identity_clause(
            2, self._icfg(worker))

    def test_an_unknown_count_keeps_the_singular_clause(self):
        """Without insightface, behave exactly as the version that shipped."""
        worker = self._worker()
        assert "the person's face" in worker.select_identity_clause(
            None, self._icfg(worker))

    def test_an_undetectable_photo_reports_unknown_not_zero(self, tmp_path):
        """The two must stay distinguishable — that conflation IS the bug."""
        worker = self._worker()
        assert worker._face_count(str(tmp_path / "does-not-exist.jpg")) is None


class TestKeptOutputs:
    """The worker writes into a TemporaryDirectory that is deleted the moment the
    bytes are read. When the group said an edit came out badly there was nothing
    left to look at: no output, and no record of the prompt that produced it."""

    @staticmethod
    def _png(directory, name="out.png"):
        from PIL import Image

        path = directory / name
        Image.new("RGB", (8, 8)).save(path)
        return path

    def test_the_render_and_its_sidecar_are_written(self, tmp_path):
        out = self._png(tmp_path)
        imagegen._keep_output(
            _config(keep_outputs=5, keep_outputs_dir=str(tmp_path / "log")),
            out, {"mode": "edit", "seed": 7, "editor": "flux2-klein",
                  "prompt": "Replace the cans."})
        log = tmp_path / "log"
        sidecar = next(log.glob("*.json"))
        assert sidecar.with_suffix(".png").exists()
        assert json.loads(sidecar.read_text())["editor"] == "flux2-klein"

    def test_it_is_off_by_default(self, tmp_path):
        out = self._png(tmp_path)
        target = tmp_path / "log"
        imagegen._keep_output(_config(keep_outputs_dir=str(target)), out, {})
        assert not target.exists()

    def test_only_the_last_n_survive(self, tmp_path):
        out = self._png(tmp_path)
        log = tmp_path / "log"
        for seed in range(6):
            imagegen._keep_output(
                _config(keep_outputs=3, keep_outputs_dir=str(log)),
                out, {"mode": "edit", "seed": seed})
        assert len(list(log.glob("*.png"))) == 3
        assert len(list(log.glob("*.json"))) == 3

    def test_pruning_cannot_delete_what_it_did_not_write(self, tmp_path):
        """Keyed off the sidecar, so an unrelated file in the directory is safe."""
        out = self._png(tmp_path)
        log = tmp_path / "log"
        log.mkdir()
        bystander = self._png(log, "keep-me.png")
        for seed in range(6):
            imagegen._keep_output(
                _config(keep_outputs=2, keep_outputs_dir=str(log)),
                out, {"mode": "edit", "seed": seed})
        assert bystander.exists()

    def test_a_broken_directory_never_breaks_the_render(self, tmp_path):
        """A debugging aid must not be able to lose a picture."""
        out = self._png(tmp_path)
        blocker = tmp_path / "not-a-dir"
        blocker.write_text("x")
        imagegen._keep_output(
            _config(keep_outputs=3, keep_outputs_dir=str(blocker)), out, {})
