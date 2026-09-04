"""The shipped model catalogue is a floor, not a fallback.

Includes the guard that the two shipped copies — the dict in the monolith and
the seed file that is copied into a fresh install's config/ — cannot drift apart
again. They disagreed in both directions before this: the dict alone held
large-v3-turbo (pointing at a repo Systran has since withdrawn), the seed alone
held three distil models, and they gave different sizes for a fourth.
"""

import ast
import json
import os
import re

import pytest

from stt import model_catalog as mc

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = os.path.join(REPO, "config", "faster_whisper_models.default.json")


def shipped_dict():
    """FASTER_WHISPER_MODELS, read out of the monolith without importing it."""
    src = open(os.path.join(REPO, "speech_to_text.py"), encoding="utf-8").read()
    start = src.index("FASTER_WHISPER_MODELS = {")
    end = src.index("\n}\n", start) + 2
    return ast.literal_eval(src[start + len("FASTER_WHISPER_MODELS = "):end])


def seed_dict():
    with open(SEED, encoding="utf-8") as fh:
        return json.load(fh)["models"]


class TestMergeCatalog:
    def test_a_shipped_entry_overrides_the_cached_one(self):
        """The whole point: a repo corrected in code must reach a machine that
        already has a cache file, which used to be returned unconditionally and
        for ever."""
        merged = mc.merge_catalog(
            {"turbo": {"repo": "new/turbo"}},
            {"turbo": {"repo": "withdrawn/turbo"}},
        )
        assert merged["turbo"]["repo"] == "new/turbo"

    def test_discovered_extras_survive(self):
        """A newer Hub than this release is a reason to offer more models."""
        merged = mc.merge_catalog({"a": {"repo": "r/a"}}, {"b": {"repo": "r/b"}})
        assert sorted(merged) == ["a", "b"]

    def test_a_shipped_model_cannot_be_dropped_by_discovery(self):
        """Discovery searches one author, so it would delete large-v3-turbo —
        which is no longer published by that author — on every refresh."""
        merged = mc.merge_catalog({"turbo": {"repo": "other/turbo"}}, {})
        assert "turbo" in merged

    def test_an_empty_cache_yields_the_shipped_catalogue(self):
        shipped = {"a": {"repo": "r/a"}}
        assert mc.merge_catalog(shipped, {}) == shipped

    def test_junk_cache_entries_are_dropped(self):
        """A hand-edited or truncated cache file must not reach the UI."""
        merged = mc.merge_catalog({}, {"a": "not a dict", "b": {"repo": "r/b"}})
        assert list(merged) == ["b"]

    def test_the_inputs_are_not_mutated(self):
        shipped = {"a": {"repo": "r/a"}}
        cached = {"b": {"repo": "r/b"}}
        merged = mc.merge_catalog(shipped, cached)
        merged["a"]["repo"] = "changed"
        merged["c"] = {}
        assert shipped == {"a": {"repo": "r/a"}}
        assert cached == {"b": {"repo": "r/b"}}


class TestCatalogRepos:
    def test_entries_without_a_repo_are_skipped(self):
        assert mc.catalog_repos({"a": {"repo": "r/a"}, "b": {}, "c": "junk"}) == {"a": "r/a"}


class TestShippedCatalogue:
    def test_the_dict_and_the_seed_file_agree(self):
        """Two shipped copies of the same catalogue is how one of them ended up
        holding a dead address that nobody noticed."""
        assert shipped_dict() == seed_dict()

    def test_every_entry_is_complete(self):
        for name, entry in shipped_dict().items():
            assert set(entry) == {"repo", "size", "params", "lang"}, name
            assert all(entry.values()), name

    def test_no_entry_points_at_the_withdrawn_turbo_repo(self):
        """Systran/faster-whisper-large-v3-turbo answers 401 — it was withdrawn.
        Pinned by name because the Model Manager offering an undownloadable model
        is exactly the failure this catalogue caused."""
        repos = set(mc.catalog_repos(shipped_dict()).values())
        assert "Systran/faster-whisper-large-v3-turbo" not in repos

    def test_turbo_uses_the_repo_the_library_itself_resolves(self):
        """faster_whisper.utils._MODELS maps "large-v3-turbo" to this repo, so
        it is the conversion the pinned loader would fetch for the bare name.
        Read from the installed package when it is there; skipped in the ML-free
        CI environment, where the assertion has nothing to check against."""
        utils = pytest.importorskip("faster_whisper.utils")
        expected = utils._MODELS["large-v3-turbo"]
        assert shipped_dict()["large-v3-turbo"]["repo"] == expected

    @pytest.mark.parametrize("field,pattern", [
        ("size", r"^~\d+(\.\d+)?(MB|GB)$"),
        ("params", r"^\d+M$"),
    ])
    def test_display_fields_are_well_formed(self, field, pattern):
        for name, entry in shipped_dict().items():
            assert re.match(pattern, entry[field]), f"{name}: {entry[field]}"

    def test_every_repo_id_is_owner_slash_name(self):
        for name, repo in mc.catalog_repos(shipped_dict()).items():
            assert repo.count("/") == 1 and all(repo.split("/")), name
