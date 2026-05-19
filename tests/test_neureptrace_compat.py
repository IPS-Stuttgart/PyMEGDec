from __future__ import annotations

import types

from pymegdec import neureptrace_compat


def test_migration_status_prints_known_mapping(capsys):
    assert neureptrace_compat.migration_status([], prog="pymegdec migration") == 0
    captured = capsys.readouterr()
    assert "pymegdec stimulus decoding" in captured.out
    assert "neureptrace dataset run" in captured.out


def test_deprecated_handler_emits_visible_warning(capsys, monkeypatch):
    monkeypatch.delenv(neureptrace_compat.SUPPRESS_ENV, raising=False)

    def handler(argv, prog):
        assert argv == ["--x"]
        assert prog == "legacy"
        return 0

    wrapped = neureptrace_compat.deprecated_handler(handler, "pymegdec stimulus decoding")
    assert wrapped(["--x"], "legacy") == 0
    captured = capsys.readouterr()
    assert "PyMEGDec migration warning" in captured.err
    assert "pymegdec stimulus decoding" in captured.err


def test_neureptrace_command_handler_forwards_to_neureptrace(monkeypatch):
    calls: list[list[str]] = []

    def fake_import_module(name):
        assert name == "neureptrace.cli"

        def fake_main(argv):
            calls.append(list(argv))
            return 0

        return types.SimpleNamespace(main=fake_main)

    monkeypatch.setattr(neureptrace_compat, "import_module", fake_import_module)
    handler = neureptrace_compat.neureptrace_command_handler("mne-time-decode")
    assert handler(["--epochs", "x-epo.fif"], "pymegdec mne-time-decode") == 0
    assert calls == [["mne-time-decode", "--epochs", "x-epo.fif"]]
