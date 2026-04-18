"""Smoke tests for the unified ``fly`` CLI."""

from __future__ import annotations

import pytest

from fyl.cli import build_parser, main


class TestCLIParser:
    def test_help_lists_subcommands(self, capsys):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(['--help'])
        captured = capsys.readouterr()
        assert 'mount' in captured.out
        assert 'optimize' in captured.out

    def test_optimize_invocation(self, multi_fly, fast_kdf, monkeypatch):
        alpha = multi_fly.mount('alpha')
        assert alpha.write('/f', b'hello', 0) == 5
        multi_fly.unmount_all()

        # Use our fast KDF instead of the default production one.
        monkeypatch.setattr('fyl.cli._build_kdf', lambda _args: fast_kdf)

        rc = main(['optimize', str(multi_fly.path), '--password', 'alpha'])
        assert rc == 0

        reopened = multi_fly.mount('alpha')
        assert reopened.read('/f', 5, 0) == b'hello'
