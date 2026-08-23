"""Monolith wiring around LLM translation, exercised without importing it.

speech_to_text.py cannot be imported (ML libraries, Flask app, background
threads at import time), so these extract the individual functions and exec them
against a stub namespace — see tests/conftest.py:extract_definitions.

Every case here is a defect that reached a running server. The pure logic in
stt/ was fully covered at the time; what broke was the wiring around it, so that
is what these pin down.
"""

import threading

import pytest

from conftest import extract_definitions
from stt.model_disk import dir_has_weights, model_presence
from stt.session_meta import row_label_if_changed


class _StubTorchCuda:
    def __init__(self, available):
        self._available = available

    def is_available(self):
        return self._available


class _StubTorch:
    """Stands in for the lazily-imported torch module."""

    def __init__(self, cuda=False, mps=False):
        self.cuda = _StubTorchCuda(cuda)
        self.backends = type("B", (), {"mps": type("M", (), {
            "is_available": staticmethod(lambda: mps)})()})()


class TestDeviceLabel:
    """_llm_device_label reports where the in-process GGUF actually runs.

    A paired machine records this as the device that translated its captions, so
    a wrong answer is written into a transcript.
    """

    def _label(self, *, config, local_llm, torch=None, gpu_layers=lambda v, has: -1):
        ns = extract_definitions(
            "speech_to_text.py", ["_llm_device_label"],
            extra_globals={
                "config": config,
                "_local_llm": local_llm,
                "torch": torch,
                "_lazy_import_ml_libraries": lambda: None,
                "_llm_resolve_gpu_layers": gpu_layers,
            })
        return ns["_llm_device_label"]()

    CFG_LOCAL = {"live_translation": {"llm": {"provider": "local", "n_gpu_layers": "auto"}}}

    def test_metal_when_mps_is_the_accelerator(self):
        assert self._label(config=self.CFG_LOCAL, local_llm=object(),
                           torch=_StubTorch(mps=True)) == "metal"

    def test_cuda_when_cuda_is_present(self):
        assert self._label(config=self.CFG_LOCAL, local_llm=object(),
                           torch=_StubTorch(cuda=True)) == "cuda"

    def test_cpu_when_no_layers_are_offloaded(self):
        # n_gpu_layers 0 means CPU even on a machine that has a GPU.
        assert self._label(config=self.CFG_LOCAL, local_llm=object(),
                           torch=_StubTorch(cuda=True),
                           gpu_layers=lambda v, has: 0) == "cpu"

    def test_none_until_the_model_is_loaded(self):
        assert self._label(config=self.CFG_LOCAL, local_llm=None,
                           torch=_StubTorch(mps=True)) is None

    def test_none_for_an_endpoint_provider(self):
        """The device belongs to the other machine; claiming one would be a guess."""
        cfg = {"live_translation": {"llm": {"provider": "endpoint"}}}
        assert self._label(config=cfg, local_llm=object(), torch=_StubTorch(mps=True)) is None


class TestLocalLlmLoadPreconditions:
    """get_local_llm's ordering contract.

    Two shipped defects live here. It probes torch.cuda to decide n_gpu_layers,
    but the module-level torch is None until the lazy importer runs — the first
    attempt died with "'NoneType' object has no attribute 'cuda'". And the probe
    must happen BEFORE llama_cpp is imported, because the CUDA build links
    against a runtime that torch only puts within reach once it initialises;
    importing llama_cpp first fails with "libcudart.so.12: cannot open shared
    object file" even though the library is present.
    """

    def _run(self, *, torch_module, available=True, model_path_exists=False, calls=None):
        calls = calls if calls is not None else []
        state = {"torch": torch_module}

        def lazy_import():
            calls.append("lazy_import")
            # The real importer assigns the module global; emulate that.
            ns["torch"] = _StubTorch(mps=True)

        ns = extract_definitions(
            "speech_to_text.py", ["get_local_llm"],
            extra_globals={
                "config": {"live_translation": {"llm": {
                    "provider": "local", "gguf_repo": "r/x", "gguf_file": "m.gguf",
                    "n_gpu_layers": "auto"}}},
                "_local_llm": None,
                "_local_llm_path": "",
                "_local_llm_failed": False,
                "_local_llm_lock": __import__("threading").Lock(),
                "local_llm_available": lambda: available,
                "_lazy_import_ml_libraries": lazy_import,
                "_llm_local_model_path": lambda d, r, f: "/nonexistent/m.gguf",
                "_llm_resolve_gpu_layers": lambda v, has: -1,
                "MODELS_DIR": "/models",
                "unload_local_llm": lambda: None,
                "torch": torch_module,
                "os": __import__("os"),
                # Counted where a caption actually asks for the model, so the number
                # means "captions that went out untranslated" and not "times anything
                # probed the runtime".
                "_llm_passthrough_captions": 0,
            })
        ns.setdefault("torch", torch_module)
        return ns["get_local_llm"](), calls, state

    def test_missing_runtime_is_reported_not_raised(self):
        """An absent llama-cpp-python degrades to the NMT model, like panns does."""
        result, calls, _ = self._run(torch_module=None, available=False)
        assert result is None
        assert "lazy_import" not in calls, "must not do ML work when the runtime is absent"

    def test_a_missing_model_file_returns_none_rather_than_raising(self):
        # Only the guard: the path check happens before the torch probe, so this
        # does not exercise the ordering. That is pinned structurally below,
        # because actually reaching the probe requires llama-cpp-python, which CI
        # deliberately does not install.
        result, _, _ = self._run(torch_module=None, available=True)
        assert result is None

    def test_the_gpu_probe_is_ordered_against_both_shipped_failures(self):
        """Source-level, because the runtime path needs llama-cpp-python.

        Two orderings must hold inside get_local_llm:

        1. _lazy_import_ml_libraries() before any torch.* use — the module-level
           torch is None until it runs, and the first attempt died on
           "'NoneType' object has no attribute 'cuda'".
        2. the torch probe before `from llama_cpp import Llama` — the CUDA build
           links against a runtime torch only puts within reach once it
           initialises, so importing llama_cpp first fails with
           "libcudart.so.12: cannot open shared object file".
        """
        import ast
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent / "speech_to_text.py").read_text(encoding="utf-8")
        func = next(n for n in ast.parse(src).body
                    if isinstance(n, ast.FunctionDef) and n.name == "get_local_llm")

        # Node positions, not text: the explanation of this very ordering is a
        # comment above the code and would match any textual search.
        lazy = torch_use = llama_import = None
        for node in ast.walk(func):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "_lazy_import_ml_libraries" and lazy is None):
                lazy = node.lineno
            if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                    and node.value.id == "torch" and torch_use is None):
                torch_use = node.lineno
            if (isinstance(node, ast.ImportFrom) and node.module == "llama_cpp"
                    and llama_import is None):
                llama_import = node.lineno

        assert lazy is not None, "the lazy ML importer must be called before probing torch"
        assert torch_use is not None and llama_import is not None
        assert lazy <= torch_use, "torch is None until the lazy importer has run"
        assert torch_use < llama_import, (
            "the CUDA runtime is only resolvable after torch initialises; "
            "importing llama_cpp first fails with libcudart.so.12 not found")


class TestTranslateTextEngineSelection:
    """translate_text must describe the model it was HANDED, not the live one.

    It takes model/tokenizer as parameters but read is_madlad and is_ct2 from the
    globals describing the live model. Batch file transcription loads its own, so
    a MADLAD file model was tokenized as NLLB — no "<2xx>" target prefix, meaning
    no target language reached the model at all.
    """

    MADLAD = "google/madlad400-3b-mt"
    NLLB = "facebook/nllb-200-distilled-600M"

    def _tokenize_style(self, *, live_model_id, passed_model_id, live_is_ct2=False,
                        passed_is_ct2=None):
        seen = {}

        def tokenizer(text, **kw):
            seen["text"] = text
            return {"input_ids": _FakeTensor()}
        tokenizer.src_lang = None

        ns = extract_definitions(
            "speech_to_text.py", ["translate_text"],
            extra_globals={
                "_live_translation_model_id": live_model_id,
                "_live_translation_is_ct2": live_is_ct2,
                "is_madlad_model": lambda mid: "madlad" in (mid or "").lower(),
                "build_madlad_input": lambda text, lang: f"<2{lang}> {text}",
                "NLLB_LANG_CODES": {"ru": "rus_Cyrl"},
                "_translate_text_ct2": lambda *a, **k: seen.setdefault("ct2", True) or "ct2",
                "_apply_glossary": lambda t, s, tl: t,
                "_record_local_translate_ms": lambda ms: None,
                "coerce_float": lambda v, d, **k: d,
                "coerce_int": lambda v, d, **k: d,
                "config": {"live_translation": {}},
            })
        try:
            ns["translate_text"]("Мир вам.", "ru", "en", _FakeModel(), tokenizer,
                                 model_id=passed_model_id, is_ct2=passed_is_ct2)
        except Exception:
            pass  # generation is stubbed out; only the tokenization branch matters
        return seen

    def test_passed_madlad_model_gets_the_target_prefix(self):
        seen = self._tokenize_style(live_model_id=self.NLLB, passed_model_id=self.MADLAD)
        assert seen.get("text", "").startswith("<2en>"), (
            "a MADLAD model must receive its target-language prefix even when the "
            "live model is NLLB")

    def test_passed_nllb_model_is_not_given_a_madlad_prefix(self):
        seen = self._tokenize_style(live_model_id=self.MADLAD, passed_model_id=self.NLLB)
        assert not seen.get("text", "").startswith("<2"), (
            "an NLLB model must not be tokenized as MADLAD because the live model is")

    def test_caller_can_opt_out_of_the_ct2_path(self):
        seen = self._tokenize_style(live_model_id=self.NLLB, passed_model_id=self.NLLB,
                                    live_is_ct2=True, passed_is_ct2=False)
        assert "ct2" not in seen, "a transformers model must not be sent to CTranslate2"

    def test_defaults_still_follow_the_live_globals(self):
        """The live pipeline passes nothing, so its behaviour must be unchanged."""
        seen = self._tokenize_style(live_model_id=self.MADLAD, passed_model_id=None)
        assert seen.get("text", "").startswith("<2en>")


class _FakeTensor:
    def to(self, device):
        return self


class _FakeModel:
    def parameters(self):
        raise StopIteration  # generation is not under test

    def generate(self, **kw):
        raise RuntimeError("not under test")


@pytest.mark.parametrize("name", [
    "_llm_device_label", "get_local_llm", "translate_text", "_translate_via_llm",
])
def test_the_functions_under_test_still_exist(name):
    """A rename would otherwise turn these tests into silent no-ops."""
    extract_definitions("speech_to_text.py", [name], extra_globals={})


class TestLlmFallbackSelector:
    """llm.fallback decides whether the NMT weights are ever loaded at all.

    On a 16 GB translation box those weights are several GB held to serve about one
    caption in a hundred, and they are the room a larger LLM needs.
    """

    def ns(self, cfg):
        # Wired to the real stt.llm_translate resolver, not a stand-in: the point of
        # the wrapper is that the caption path and the settings route resolve the
        # value identically, and a stub here would let the two drift apart unseen.
        from stt import llm_translate as L
        return extract_definitions("speech_to_text.py", ["_llm_fallback_is_skip"],
                                   {"config": cfg,
                                    "_llm_resolve_fallback": L.resolve_fallback,
                                    "_LLM_FALLBACK_SKIP": L.FALLBACK_SKIP})

    def test_default_keeps_the_nmt_fallback(self):
        # Absent config must not silently stop translating a declined caption.
        assert self.ns({})["_llm_fallback_is_skip"]() is False
        assert self.ns({"live_translation": {}})["_llm_fallback_is_skip"]() is False
        assert self.ns({"live_translation": {"llm": {}}})["_llm_fallback_is_skip"]() is False

    def test_skip_is_recognised(self):
        cfg = {"live_translation": {"llm": {"fallback": "skip"}}}
        assert self.ns(cfg)["_llm_fallback_is_skip"]() is True

    def test_case_and_padding_do_not_matter(self):
        cfg = {"live_translation": {"llm": {"fallback": "  SKIP "}}}
        assert self.ns(cfg)["_llm_fallback_is_skip"]() is True

    def test_an_explicit_nmt_or_an_unknown_word_keeps_the_fallback(self):
        for value in ("nmt", "local", "", None, "yes"):
            cfg = {"live_translation": {"llm": {"fallback": value}}}
            assert self.ns(cfg)["_llm_fallback_is_skip"]() is False, value


class TestTranslationReadyWithLlm:
    """The persistence gate must follow the engine actually producing captions.

    is_live_translation_ready() reports the NMT model's load state. With
    llm.fallback = "skip" that model never loads, so asking it would report "not
    ready" for a whole service and every translated caption would be produced and
    then dropped instead of saved.
    """

    def ns(self, cfg, nmt_loaded=False, llm_loaded=False):
        return extract_definitions(
            "speech_to_text.py", ["is_live_translation_ready"],
            {"config": cfg, "_live_translation_model_loaded": nmt_loaded,
             "is_local_llm_loaded": lambda: llm_loaded})

    def test_offloading_is_ready_regardless(self):
        cfg = {"live_translation": {"remote": {"enabled": True, "endpoint": "http://h"}}}
        assert self.ns(cfg)["is_live_translation_ready"]() is True

    def test_local_llm_gates_on_the_gguf_not_the_nmt_model(self):
        cfg = {"live_translation": {"translation_method": "llm",
                                    "llm": {"provider": "local"}}}
        assert self.ns(cfg, llm_loaded=True)["is_live_translation_ready"]() is True
        assert self.ns(cfg, llm_loaded=False)["is_live_translation_ready"]() is False

    def test_a_loaded_nmt_model_does_not_make_an_unloaded_gguf_ready(self):
        cfg = {"live_translation": {"translation_method": "llm",
                                    "llm": {"provider": "local"}}}
        ready = self.ns(cfg, nmt_loaded=True, llm_loaded=False)["is_live_translation_ready"]()
        assert ready is False, "captions would be persisted while the GGUF still echoes the source"

    def test_endpoint_provider_is_ready_once_it_is_configured(self):
        cfg = {"live_translation": {"translation_method": "llm",
                                    "llm": {"provider": "endpoint", "endpoint": "http://h",
                                            "model": "m"}}}
        assert self.ns(cfg)["is_live_translation_ready"]() is True

    def test_an_unconfigured_endpoint_is_not_ready(self):
        for llm in ({"provider": "endpoint", "endpoint": "", "model": "m"},
                    {"provider": "endpoint", "endpoint": "http://h", "model": ""}):
            cfg = {"live_translation": {"translation_method": "llm", "llm": llm}}
            assert self.ns(cfg)["is_live_translation_ready"]() is False, llm

    def test_the_nmt_path_is_unchanged(self):
        cfg = {"live_translation": {"translation_method": "madlad"}}
        assert self.ns(cfg, nmt_loaded=True)["is_live_translation_ready"]() is True
        assert self.ns(cfg, nmt_loaded=False)["is_live_translation_ready"]() is False


class TestLocalFallbackReady:
    """Whether "fall back to local translation" can actually be honoured.

    The setting quietly becomes "skip translation" when no local model is on
    disk, and both produce the same thing on screen: untranslated captions. The
    operator picks one behaviour and silently gets the other, during a service.
    This is what lets the page say so beforehand.
    """

    def ns(self, cfg, models_dir, gguf_path=""):
        return extract_definitions(
            "speech_to_text.py", ["_local_fallback_ready"],
            {"config": cfg, "MODELS_DIR": str(models_dir),
             "dir_has_weights": dir_has_weights,
             "model_presence": model_presence,
             "_resolve_live_translation_model_id": lambda lt: lt.get("translation_model", ""),
             "_llm_local_model_path": lambda d, repo, f: gguf_path})

    def nmt(self, tmp_path, model_id, downloaded):
        if downloaded:
            d = tmp_path / model_id.replace("/", "--")
            d.mkdir(parents=True)
            (d / "model.safetensors").write_text("x")
        return {"live_translation": {"translation_method": "nllb", "translation_model": model_id}}

    def test_a_downloaded_nmt_model_is_ready(self, tmp_path):
        cfg = self.nmt(tmp_path, "facebook/nllb-200-distilled-600M", True)
        assert self.ns(cfg, tmp_path)["_local_fallback_ready"]() is True

    def test_a_missing_nmt_model_is_not_ready(self, tmp_path):
        cfg = self.nmt(tmp_path, "facebook/nllb-200-distilled-600M", False)
        assert self.ns(cfg, tmp_path)["_local_fallback_ready"]() is False

    def test_a_directory_without_weights_is_not_ready(self, tmp_path):
        # A part-finished or cancelled download leaves the directory behind.
        (tmp_path / "facebook--nllb-200-distilled-600M").mkdir(parents=True)
        cfg = self.nmt(tmp_path, "facebook/nllb-200-distilled-600M", False)
        assert self.ns(cfg, tmp_path)["_local_fallback_ready"]() is False

    def test_a_ctranslate2_conversion_is_ready(self, tmp_path):
        # Converting for CT2 and then reclaiming the HF weights leaves the
        # tokenizer behind and the weights in a sibling directory. That machine
        # translates fine, and used to be reported as having no model at all.
        hf = tmp_path / "facebook--nllb-200-distilled-600M"
        hf.mkdir(parents=True)
        (hf / "config.json").write_text("{}")
        ct2 = tmp_path / "facebook--nllb-200-distilled-600M-ct2-int8"
        ct2.mkdir(parents=True)
        (ct2 / "model.bin").write_text("x")

        cfg = self.nmt(tmp_path, "facebook/nllb-200-distilled-600M", False)
        assert self.ns(cfg, tmp_path)["_local_fallback_ready"]() is True

    def test_no_model_configured_is_not_ready(self, tmp_path):
        cfg = {"live_translation": {"translation_method": "nllb", "translation_model": ""}}
        assert self.ns(cfg, tmp_path)["_local_fallback_ready"]() is False

    @pytest.mark.parametrize("method", ["whisper_translate", "whisper_forced_lang"])
    def test_whisper_translate_needs_no_separate_model(self, tmp_path, method):
        # The ASR pass produces the translation; there is nothing else to download.
        cfg = {"live_translation": {"translation_method": method}}
        assert self.ns(cfg, tmp_path)["_local_fallback_ready"]() is True

    def test_a_local_llm_is_ready_when_its_gguf_exists(self, tmp_path):
        gguf = tmp_path / "m.gguf"
        gguf.write_text("x")
        cfg = {"live_translation": {"translation_method": "llm",
                                    "llm": {"provider": "local", "gguf_repo": "r", "gguf_file": "m.gguf"}}}
        assert self.ns(cfg, tmp_path, gguf_path=str(gguf))["_local_fallback_ready"]() is True

    def test_a_local_llm_without_its_gguf_is_not_ready(self, tmp_path):
        cfg = {"live_translation": {"translation_method": "llm",
                                    "llm": {"provider": "local", "gguf_repo": "r", "gguf_file": "m.gguf"}}}
        assert self.ns(cfg, tmp_path, gguf_path=str(tmp_path / "absent.gguf"))["_local_fallback_ready"]() is False

    def test_an_endpoint_llm_is_ready_once_configured(self, tmp_path):
        cfg = {"live_translation": {"translation_method": "llm",
                                    "llm": {"provider": "endpoint", "endpoint": "http://h"}}}
        assert self.ns(cfg, tmp_path)["_local_fallback_ready"]() is True
        cfg["live_translation"]["llm"]["endpoint"] = ""
        assert self.ns(cfg, tmp_path)["_local_fallback_ready"]() is False


class TestNoFallbackModelConfigured:
    """Selecting "None" must be a clean no-op, not an error per caption.

    An empty id used to reach load_translation_model(), where "" joined to
    MODELS_DIR is a real directory — so it loaded the models folder as a model
    and raised, once per caption and again on every preload. Guarding inside
    get_live_translation_model covers all five callers at once, including
    /api/translate/preload, which an offload client calls on every service start
    and which would otherwise load the very model the choice was avoiding.
    """

    def ns(self, model_id, loader_calls, loaded=None):
        def loader(*args, **kwargs):
            loader_calls.append((args, kwargs))
            return ("model", "tokenizer")

        return extract_definitions(
            "speech_to_text.py", ["get_live_translation_model"],
            {"config": {"live_translation": {}},
             "_live_translation_lock": threading.Lock(),
             "_live_translation_model": loaded,
             "_live_translation_tokenizer": None,
             "_live_translation_model_loaded": False,
             "_live_translation_model_loading": False,
             "_live_translation_model_id": None,
             "_live_translation_model_wanted": False,
             "_live_translation_is_ct2": False,
             "_live_translation_device": None,
             "_no_fallback_model_logged": False,
             "_ts_get": lambda k, d=None: d,
             "load_translation_model": loader,
             "_warmup_translation_model": lambda *a, **k: None,
             "_record_session_meta_change": lambda *a, **k: None,
             "make_db_world_readable": lambda *a: None})

    @pytest.mark.parametrize("model_id", ["", "   ", None])
    def test_no_model_returns_nothing_and_loads_nothing(self, model_id):
        calls = []
        ns = self.ns(model_id, calls)
        assert ns["get_live_translation_model"](True, model_id=model_id) == (None, None)
        assert calls == [], "an empty model id must never reach the loader"

    def test_a_configured_model_still_loads(self):
        # The regression pin: every existing install goes down this path.
        calls = []
        ns = self.ns("facebook/nllb-200-distilled-600M", calls)
        model, tokenizer = ns["get_live_translation_model"](
            True, model_id="facebook/nllb-200-distilled-600M")
        assert (model, tokenizer) == ("model", "tokenizer")
        assert len(calls) == 1


class TestMtBaselineEstablishesItself:
    """Which caption carries the model name, and which are left NULL.

    The baseline used to be set in initialize_database — the spawned transcription
    worker — while the rows are written by the web process. A global set in one is
    invisible in the other, so the web process saw an empty baseline and repeated
    the label on every row. Safe, but not the deduplication it was meant to be, and
    only visible after a real service. Deriving it from the first caption needs no
    cross-process channel and says the same thing.
    """

    def ns(self, baseline="", label="192.168.2.52:8080 (gemma.gguf)"):
        state = {"value": baseline}
        recorded = {}

        class Local:
            pass

        return extract_definitions(
            "speech_to_text.py", ["_record_mt_engine", "_set_mt_baseline_label"],
            {"config": {"live_translation": {}},
             "_mt_provenance": Local(),
             "_mt_baseline_label": state,
             "_session_mt_row_label": lambda *a, **k: label,
             "_session_row_label_if_changed": row_label_if_changed,
             "_remote_effective_status": lambda: None,
             "_recorded": recorded}), state

    def test_the_first_caption_carries_the_label(self):
        ns, state = self.ns()
        ns["_record_mt_engine"]("remote")
        assert ns["_mt_provenance"].model == "192.168.2.52:8080 (gemma.gguf)"
        assert state["value"] == "192.168.2.52:8080 (gemma.gguf)", "and becomes the baseline"

    def test_every_identical_caption_after_it_is_null(self):
        ns, _ = self.ns()
        ns["_record_mt_engine"]("remote")
        for _ in range(5):
            ns["_record_mt_engine"]("remote")
            assert ns["_mt_provenance"].model is None

    def test_a_changed_model_reasserts_itself(self):
        ns, _state = self.ns(baseline="192.168.2.52:8080 (old.gguf)")
        ns["_record_mt_engine"]("remote")
        assert ns["_mt_provenance"].model == "192.168.2.52:8080 (gemma.gguf)"

    def test_clearing_the_baseline_starts_a_new_session(self):
        # start_transcription resets it, so one session's model cannot become the
        # next one's baseline and silently make a changed setup look unchanged.
        ns, _state = self.ns()
        ns["_record_mt_engine"]("remote")
        assert ns["_record_mt_engine"]("remote") == "remote"
        assert ns["_mt_provenance"].model is None
        ns["_set_mt_baseline_label"]("")
        ns["_record_mt_engine"]("remote")
        assert ns["_mt_provenance"].model == "192.168.2.52:8080 (gemma.gguf)"

    def test_an_empty_label_never_becomes_the_baseline(self):
        ns, state = self.ns(label="")
        ns["_record_mt_engine"]("none")
        assert ns["_mt_provenance"].model is None
        assert state["value"] == "", "an untranslated caption must not fix the baseline"


class TestPreloadWarmsTheRightEngine:
    """/api/translate/preload runs on the offload server when its client starts.

    It loaded the NMT model unconditionally — on an LLM box that is the fallback,
    several GB for a model serving about one caption in a hundred, while the model
    doing the work stayed cold. Loading is lazy, so the cost landed on whoever
    spoke first: a measured service opened with six untranslated captions, 56
    seconds, because the GGUF only began loading when the first caption arrived.
    """

    def call(self, live_translation):
        loaded = {"nmt": 0, "llm": 0}

        class Thread:
            def __init__(self, target=None, daemon=None):
                self._t = target

            def start(self):
                self._t()

        ns = extract_definitions(
            "speech_to_text.py", ["translate_preload"],
            {"config": {"live_translation": live_translation},
             "request": type("R", (), {"remote_addr": "192.168.2.62"})(),
             "jsonify": lambda p: p,
             "_paired_client_ok": lambda ip=None: True,
             "get_live_translation_model": lambda *a, **k: loaded.__setitem__("nmt", loaded["nmt"] + 1),
             "_warm_local_llm": lambda *a, **k: loaded.__setitem__("llm", loaded["llm"] + 1),
             "threading": type("T", (), {"Thread": Thread}),
             "is_local_llm_loaded": lambda: False,
             "is_live_translation_model_loaded": lambda: False,
             "is_live_translation_model_loading": lambda: False,
             "_live_translation_model_wanted": False})
        return ns["translate_preload"](), loaded

    LLM_LOCAL = {"enabled": True, "translation_method": "llm",
                 "llm": {"provider": "local", "gguf_file": "m.gguf"}}

    def test_an_llm_box_warms_the_llm_and_not_the_fallback(self):
        body, loaded = self.call(self.LLM_LOCAL)
        assert loaded == {"nmt": 0, "llm": 1}
        assert body["engine"] == "llm"

    def test_an_endpoint_llm_loads_nothing_locally(self):
        body, loaded = self.call({"enabled": True, "translation_method": "llm",
                                  "llm": {"provider": "endpoint", "endpoint": "http://h"}})
        assert loaded == {"nmt": 0, "llm": 0}
        assert body["engine"] == "llm" and body["loaded"] is False
        assert body["success"] is True, "nothing to load is not a failure"

    @pytest.mark.parametrize("method", ["whisper_translate", "whisper_forced_lang"])
    def test_whisper_has_nothing_to_preload(self, method):
        body, loaded = self.call({"enabled": True, "translation_method": method})
        assert loaded == {"nmt": 0, "llm": 0}
        assert body["engine"] == "none"

    @pytest.mark.parametrize("method", ["nllb", "madlad"])
    def test_an_nmt_box_still_loads_its_model(self, method):
        # The regression pin: every existing offload server goes down this path.
        body, loaded = self.call({"enabled": True, "translation_method": method,
                                  "translation_model": "facebook/nllb-200-distilled-600M"})
        assert loaded == {"nmt": 1, "llm": 0}
        assert body["engine"] == "nmt"

    def test_translation_disabled_is_refused(self):
        body, loaded = self.call({"enabled": False})
        assert loaded == {"nmt": 0, "llm": 0}
        assert body["success"] is False

    def test_a_resident_fallback_does_not_suppress_the_warm_up(self):
        """The NMT-loaded check describes the fallback, not the engine in use.

        Asked first, it answered "already loaded" on an LLM box and warmed
        nothing — leaving the model that does the work cold.
        """
        loaded = {"nmt": 0, "llm": 0}

        class Thread:
            def __init__(self, target=None, daemon=None):
                self._t = target

            def start(self):
                self._t()

        ns = extract_definitions(
            "speech_to_text.py", ["translate_preload"],
            {"config": {"live_translation": self.LLM_LOCAL},
             "request": type("R", (), {"remote_addr": "192.168.2.62"})(),
             "jsonify": lambda p: p,
             "_paired_client_ok": lambda ip=None: True,
             "get_live_translation_model": lambda *a, **k: loaded.__setitem__("nmt", loaded["nmt"] + 1),
             "_warm_local_llm": lambda *a, **k: loaded.__setitem__("llm", loaded["llm"] + 1),
             "threading": type("T", (), {"Thread": Thread}),
             "is_local_llm_loaded": lambda: False,
             "is_live_translation_model_loaded": lambda: True,   # the fallback IS resident
             "is_live_translation_model_loading": lambda: False,
             "_live_translation_model_wanted": False})
        body = ns["translate_preload"]()
        assert loaded == {"nmt": 0, "llm": 1}
        assert body["engine"] == "llm"


class TestPromptStyleWiring:
    """_translate_via_llm branches on the shape of the configured model.

    A TranslateGemma is not an instruction model: it takes four fields and no
    prompt. Sending it the chat prompt would put the terminology instructions
    themselves on the screen, translated — and every rule that hangs off the system
    prompt (the input budget's reservation, the corrective retry) has to agree with
    that same decision or the caption is sized and retried against a prompt that was
    never sent.
    """

    def ns(self, llm_cfg, audio_language="ru"):
        from stt import llm_translate as L

        calls = []

        def _local(text, system_prompt, max_tokens, override=None, raw_prompt=None):
            calls.append({"text": text, "system_prompt": system_prompt,
                          "raw_prompt": raw_prompt})
            return "Peace be with you."

        ns = extract_definitions(
            "speech_to_text.py", ["_translate_via_llm"],
            {"config": {"live_translation": {"llm": llm_cfg},
                        "audio": {"language": audio_language}},
             "TRANSLATION_LANGUAGES": {"en": "English", "ru": "Russian"},
             "coerce_int": lambda v, d, lo=None, hi=None: int(v) if v not in (None, "") else d,
             "coerce_float": lambda v, d, lo=None, hi=None: float(v) if v not in (None, "") else d,
             "_DEFAULT_LLM_SYSTEM_PROMPT": L.DEFAULT_SYSTEM_PROMPT_TEMPLATE,
             "_LLM_STYLE_TRANSLATEGEMMA": L.PROMPT_STYLE_TRANSLATEGEMMA,
             "_LLM_RETRY_MIN_SECONDS": 1.5,
             "_llm_prompt_style": L.resolve_prompt_style,
             "_llm_uses_system_prompt": L.uses_system_prompt,
             "_llm_system_prompt": L.build_system_prompt,
             "_llm_tg_prompt": L.build_translategemma_prompt,
             "_llm_tg_messages": L.build_translategemma_messages,
             "_llm_check": L.check_translation,
             "_llm_retry_prompt": L.retry_system_prompt,
             "_llm_retry_enabled": lambda cfg: True,
             "_llm_budget_for": lambda cfg, prompt, mt: (4096, None),
             "_llm_input_fits": lambda text, budget, counter=None: True,
             "_translate_via_local_llm": _local,
             "_calls": calls})
        return ns, calls

    LOCAL_TG = {"provider": "local", "gguf_repo": "mradermacher/translategemma-12b-it-GGUF",
                "gguf_file": "translategemma-12b-it.Q4_K_M.gguf"}
    LOCAL_CHAT = {"provider": "local", "gguf_repo": "unsloth/gemma-4-12B-it-GGUF",
                  "gguf_file": "gemma-4-12b-it-Q4_K_M.gguf"}

    def test_a_chat_model_gets_the_system_prompt_and_no_raw_prompt(self):
        ns, calls = self.ns(dict(self.LOCAL_CHAT))
        assert ns["_translate_via_llm"]("Мир вам.", "ru", "en") == "Peace be with you."
        assert calls[0]["raw_prompt"] is None
        assert "church service" in calls[0]["system_prompt"]

    def test_a_translategemma_gets_the_field_prompt_and_no_system_prompt(self):
        ns, calls = self.ns(dict(self.LOCAL_TG))
        assert ns["_translate_via_llm"]("Мир вам.", "ru", "en") == "Peace be with you."
        assert calls[0]["system_prompt"] == ""
        assert calls[0]["raw_prompt"].endswith(
            "type:text,source_lang_code:ru,target_lang_code:en,text:")

    def test_an_explicit_style_overrides_the_model_name(self):
        cfg = dict(self.LOCAL_CHAT)
        cfg["prompt_style"] = "translategemma"
        ns, calls = self.ns(cfg)
        ns["_translate_via_llm"]("Мир вам.", "ru", "en")
        assert calls[0]["raw_prompt"] is not None

    def test_auto_source_language_is_resolved_before_the_fields_are_built(self):
        # Every other engine reads "auto" as "work it out"; TranslateGemma names the
        # source explicitly, and /api/translate reaches here with "auto" in hand.
        ns, calls = self.ns(dict(self.LOCAL_TG), audio_language="ru")
        ns["_translate_via_llm"]("Мир вам.", "auto", "en")
        assert "source_lang_code:ru" in calls[0]["raw_prompt"]

    def test_an_unresolvable_source_omits_the_field_rather_than_guessing(self):
        ns, calls = self.ns(dict(self.LOCAL_TG), audio_language="auto")
        ns["_translate_via_llm"]("Мир вам.", "auto", "en")
        assert "source_lang_code" not in calls[0]["raw_prompt"]

    def test_a_rejected_caption_is_retried_on_a_chat_model(self):
        ns, calls = self.ns(dict(self.LOCAL_CHAT))
        ns["_translate_via_local_llm"] = lambda *a, **k: None
        # A rejection the retry table has a note for: numbers lost from a reference.
        ns["_translate_via_local_llm"] = self._rejecting(calls)
        ns["_translate_via_llm"]("1 Фессалоникийцам 5 глава.", "ru", "en")
        assert len(calls) == 2, "the chat style names the broken rule and asks again"
        assert "previous answer" in calls[1]["system_prompt"]

    def test_a_rejected_caption_is_not_retried_on_a_translategemma(self):
        # There is no system prompt to name the broken rule in, so a second call
        # would spend a caption's latency to get the same answer back.
        ns, calls = self.ns(dict(self.LOCAL_TG))
        ns["_translate_via_local_llm"] = self._rejecting(calls)
        assert ns["_translate_via_llm"]("1 Фессалоникийцам 5 глава.", "ru", "en") is None
        assert len(calls) == 1

    @staticmethod
    def _rejecting(calls):
        def _local(text, system_prompt, max_tokens, override=None, raw_prompt=None):
            calls.append({"text": text, "system_prompt": system_prompt,
                          "raw_prompt": raw_prompt})
            # The measured failure: a reference answered with a different passage.
            return "1 Corinthians 11:1-24 — and the recited text of the passage."
        return _local


class TestPassthroughAccounting:
    """What the operator is told when the runtime the config asks for is gone.

    The incident: a rebuilt venv dropped llama-cpp-python mid-service, get_local_llm
    returned None for every caption, fallback="skip" sent the source text out with HTTP
    200, and the only trace was one identical log line per caption. The count is what
    turns that into a number someone can act on, so it has to count captions — not the
    probes the status poll makes every five seconds.
    """

    def _namespace(self, available):
        return extract_definitions(
            "speech_to_text.py", ["get_local_llm"],
            extra_globals={
                "config": {"live_translation": {"llm": {
                    "provider": "local", "gguf_repo": "r/x", "gguf_file": "m.gguf"}}},
                "_local_llm": None,
                "_local_llm_path": "",
                "_local_llm_failed": False,
                "_local_llm_lock": threading.Lock(),
                "local_llm_available": lambda: available,
                "_lazy_import_ml_libraries": lambda: None,
                "_llm_local_model_path": lambda d, r, f: "/nonexistent/m.gguf",
                "_llm_resolve_gpu_layers": lambda v, has: -1,
                "MODELS_DIR": "/models",
                "unload_local_llm": lambda: None,
                "torch": None,
                "os": __import__("os"),
                "_llm_passthrough_captions": 0,
            })

    def test_each_untranslated_caption_is_counted(self):
        ns = self._namespace(available=False)
        for _ in range(3):
            assert ns["get_local_llm"]() is None
        assert ns["_llm_passthrough_captions"] == 3

    def test_nothing_is_counted_while_the_runtime_is_there(self):
        ns = self._namespace(available=True)
        ns["get_local_llm"]()   # fails later, on the missing model file
        assert ns["_llm_passthrough_captions"] == 0
