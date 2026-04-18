"""Smoke tests for the unified ``stashfs`` CLI."""

from __future__ import annotations

import pytest

from stashfs.cli import build_parser, main


class TestCLIParser:
    def test_help_lists_subcommands(self, capsys):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(['--help'])
        captured = capsys.readouterr()
        assert 'mount' in captured.out
        assert 'optimize' in captured.out

    def test_optimize_invocation(self, multi_stash, fast_kdf, monkeypatch):
        alpha = multi_stash.mount('alpha')
        assert alpha.write('/f', b'hello', 0) == 5
        multi_stash.unmount_all()

        # Use our fast KDF instead of the default production one.
        monkeypatch.setattr('stashfs.cli._build_kdf', lambda _args: fast_kdf)

        rc = main(['optimize', str(multi_stash.path), '--password', 'alpha'])
        assert rc == 0

        reopened = multi_stash.mount('alpha')
        assert reopened.read('/f', 5, 0) == b'hello'


class TestOptimizeCLIPasswordless:
    """``stashfs optimize <file>`` must not prompt for any password.

    Password-free optimize is the whole point of the allocation-layer
    change; the CLI is the user-visible surface of that promise.
    """

    def test_no_prompt_on_fresh_cover(self, tmp_path, fast_kdf, monkeypatch):
        cover = tmp_path / 'cover.png'
        cover.write_bytes(b'\x89PNG\r\n\x1a\n' + b'cover-bytes-' * 40)

        def must_not_prompt(_msg=''):
            raise AssertionError('getpass must not be called under plain optimize')

        monkeypatch.setattr('stashfs.cli.getpass.getpass', must_not_prompt)
        monkeypatch.setattr('stashfs.cli._build_kdf', lambda _args: fast_kdf)

        rc = main(['optimize', str(cover)])
        assert rc == 0

    def test_no_prompt_on_populated_file(self, multi_stash, fast_kdf, monkeypatch):
        alpha = multi_stash.mount('alpha')
        assert alpha.write('/f', b'hello', 0) == 5
        multi_stash.unmount_all()

        def must_not_prompt(_msg=''):
            raise AssertionError('getpass must not be called under plain optimize')

        monkeypatch.setattr('stashfs.cli.getpass.getpass', must_not_prompt)
        monkeypatch.setattr('stashfs.cli._build_kdf', lambda _args: fast_kdf)

        rc = main(['optimize', str(multi_stash.path)])
        assert rc == 0

        # File still readable under its password.
        reopened = multi_stash.mount('alpha')
        assert reopened.read('/f', 5, 0) == b'hello'


class TestImplicitMount:
    """`stash <existing-file>` should behave like `stash mount <existing-file>`.

    A single path argument that points at an existing file is the
    overwhelmingly common case; making the user type ``mount`` every
    time is friction.
    """

    def test_bare_existing_path_dispatches_to_mount(self, tmp_path, monkeypatch):
        backing = tmp_path / 'backing'
        backing.touch()

        seen: dict[str, object] = {}

        def fake_run_mount(args):
            seen['fname'] = args.fname
            seen['command'] = args.command

        monkeypatch.setattr('stashfs.cli._run_mount', lambda args: fake_run_mount(args) or 0)

        rc = main([str(backing)])
        assert rc == 0
        assert seen['command'] == 'mount'
        assert seen['fname'] == backing.resolve()

    def test_bare_nonexistent_path_still_errors_cleanly(self, tmp_path, capsys):
        missing = tmp_path / 'does-not-exist'
        with pytest.raises(SystemExit):
            main([str(missing)])
        captured = capsys.readouterr()
        # argparse prints the usage/error on stderr when no subcommand matched.
        assert 'mount' in captured.err or 'invalid' in captured.err.lower() or missing.name in captured.err

    def test_explicit_mount_still_works(self, tmp_path, monkeypatch):
        backing = tmp_path / 'backing'
        backing.touch()

        called = {'n': 0}
        monkeypatch.setattr('stashfs.cli._run_mount', lambda args: called.update(n=called['n'] + 1) or 0)

        rc = main(['mount', str(backing)])
        assert rc == 0
        assert called['n'] == 1
