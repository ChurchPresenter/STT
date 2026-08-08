"""Backups of a small config file that gets overwritten wholesale.

The case that motivated this: the glossary is one textarea and one Save button,
a save with the box empty replaced the live file with ``{"glossary": {}}``, and
nothing on the machine held the previous content.
"""

import os

import pytest

from stt import file_backup


def _write(path, text):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


def _clock_from(times):
    """A clock that returns each of ``times`` in turn, then repeats the last."""
    remaining = list(times)
    return lambda: remaining.pop(0) if len(remaining) > 1 else remaining[0]


def test_backs_up_the_previous_contents(tmp_path):
    target = _write(str(tmp_path / "glossary.json"), '{"glossary": {"a": "b"}}')

    backup = file_backup.backup_file(target)

    assert backup is not None
    with open(backup, encoding="utf-8") as handle:
        assert handle.read() == '{"glossary": {"a": "b"}}'


def test_backup_survives_the_overwrite_that_follows(tmp_path):
    """The whole point: the copy still holds the good content afterwards."""
    target = _write(str(tmp_path / "glossary.json"), '{"glossary": {"Allah": "Lord"}}')

    backup = file_backup.backup_file(target)
    _write(target, '{"glossary": {}}')

    with open(target, encoding="utf-8") as handle:
        assert handle.read() == '{"glossary": {}}'
    with open(backup, encoding="utf-8") as handle:
        assert handle.read() == '{"glossary": {"Allah": "Lord"}}'


def test_no_backup_when_the_file_does_not_exist_yet(tmp_path):
    assert file_backup.backup_file(str(tmp_path / "absent.json")) is None


def test_backup_lands_beside_the_original(tmp_path):
    target = _write(str(tmp_path / "glossary.json"), "{}")

    backup = file_backup.backup_file(target)

    assert os.path.dirname(backup) == os.path.dirname(target)
    assert os.path.basename(backup).startswith("glossary.json" + file_backup.BACKUP_INFIX)


def test_keeps_only_the_newest_versions(tmp_path):
    target = str(tmp_path / "glossary.json")
    # One save per minute so every version gets its own stamp.
    clock = _clock_from([1_700_000_000 + 60 * i for i in range(8)])
    for i in range(8):
        _write(target, f'{{"v": {i}}}')
        file_backup.backup_file(target, keep=3, clock=clock)

    backups = file_backup.existing_backups(target)
    assert len(backups) == 3
    kept = []
    for path in backups:
        with open(path, encoding="utf-8") as handle:
            kept.append(handle.read())
    assert kept == ['{"v": 5}', '{"v": 6}', '{"v": 7}']


def test_two_saves_in_one_second_keep_the_earlier_version(tmp_path):
    """A panicked double-save must not overwrite the copy worth having."""
    target = str(tmp_path / "glossary.json")
    frozen = lambda: 1_700_000_000  # noqa: E731 - one expression, clearer inline

    _write(target, '{"glossary": {"good": "terms"}}')
    first = file_backup.backup_file(target, clock=frozen)
    _write(target, '{"glossary": {}}')
    second = file_backup.backup_file(target, clock=frozen)

    assert first == second
    with open(first, encoding="utf-8") as handle:
        assert handle.read() == '{"glossary": {"good": "terms"}}'


def test_existing_backups_are_oldest_first(tmp_path):
    target = str(tmp_path / "glossary.json")
    clock = _clock_from([1_700_000_000 + 60 * i for i in range(4)])
    for i in range(4):
        _write(target, f'{{"v": {i}}}')
        file_backup.backup_file(target, clock=clock)

    backups = file_backup.existing_backups(target)
    assert backups == sorted(backups)


def test_existing_backups_ignores_unrelated_files(tmp_path):
    target = _write(str(tmp_path / "glossary.json"), "{}")
    _write(str(tmp_path / "other.json"), "{}")
    _write(str(tmp_path / "glossary.json.tmp"), "{}")
    file_backup.backup_file(target)

    assert len(file_backup.existing_backups(target)) == 1


def test_existing_backups_on_a_missing_directory_is_empty(tmp_path):
    assert file_backup.existing_backups(str(tmp_path / "nope" / "glossary.json")) == []


def test_prune_with_keep_zero_removes_every_backup(tmp_path):
    target = str(tmp_path / "glossary.json")
    clock = _clock_from([1_700_000_000 + 60 * i for i in range(3)])
    for i in range(3):
        _write(target, f'{{"v": {i}}}')
        file_backup.backup_file(target, clock=clock)

    removed = file_backup.prune(target, keep=0)

    assert len(removed) == 3
    assert file_backup.existing_backups(target) == []


def test_a_backup_that_cannot_be_written_does_not_fail_the_save(tmp_path):
    """The safety net failing must never be what stops the save going through."""
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")
    directory = tmp_path / "locked"
    directory.mkdir()
    target = _write(str(directory / "glossary.json"), '{"glossary": {}}')
    os.chmod(directory, 0o500)  # readable, not writable
    try:
        assert file_backup.backup_file(target) is None
    finally:
        os.chmod(directory, 0o700)


def test_a_backup_that_cannot_be_deleted_is_skipped_quietly(tmp_path):
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")
    directory = tmp_path / "locked"
    directory.mkdir()
    target = str(directory / "glossary.json")
    clock = _clock_from([1_700_000_000 + 60 * i for i in range(2)])
    for i in range(2):
        _write(target, f'{{"v": {i}}}')
        file_backup.backup_file(target, clock=clock)
    os.chmod(directory, 0o500)
    try:
        assert file_backup.prune(target, keep=1) == []
    finally:
        os.chmod(directory, 0o700)


def test_prune_keeps_everything_when_under_the_limit(tmp_path):
    target = _write(str(tmp_path / "glossary.json"), "{}")
    file_backup.backup_file(target)

    assert file_backup.prune(target, keep=5) == []
    assert len(file_backup.existing_backups(target)) == 1
