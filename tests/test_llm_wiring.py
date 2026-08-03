"""Monolith wiring around LLM translation, exercised without importing it.

speech_to_text.py cannot be imported (ML libraries, Flask app, background
threads at import time), so these extract the individual functions and exec them
against a stub namespace — see tests/conftest.py:extract_definitions.

Every case here is a defect that reached a running server. The pure logic in
stt/ was fully covered at the time; what broke was the wiring around it, so that
is what these pin down.
"""

import pytest

from conftest import extract_definitions


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
        return extract_definitions("speech_to_text.py", ["_llm_fallback_is_skip"],
                                   {"config": cfg})

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
