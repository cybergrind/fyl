"""Offline compaction for stashfs backing files.

Every mutation in ``Volume`` marks superseded chunks DEAD in the
plaintext allocation table, then appends fresh live chunks. Because
the allocation is plaintext, ``optimize`` can tell live from dead
**without any password** and reclaim the dead chunks without touching
slot wraps or file indexes — their logical chunk ids stay stable.

The source container is never mutated; the rebuild is written to
``<path>.tmp`` and atomically renamed over the source so a crash
anywhere before the rename leaves the original untouched.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from stashfs.container import FLAGS_OFFSET, N_SLOTS, Container
from stashfs.crypto import KDF
from stashfs.fuse_app import _looks_like_fuse_mount
from stashfs.slot_table import FLAG_FREE, FLAG_OCCUPIED, SlotTable
from stashfs.storage import CoverStorage, FileWrapper


@dataclass
class OptimizeReport:
    old_size: int
    new_size: int
    rebuilt_slots: list[int] = field(default_factory=list)
    dropped_slots: list[int] = field(default_factory=list)

    @property
    def reclaimed(self) -> int:
        return self.old_size - self.new_size


class OptimizeError(Exception):
    """Raised when optimize refuses to run (live mount, locked + drop-locked disagreement, ...)."""


def optimize(
    path: Path,
    passwords: Sequence[str] = (),
    *,
    kdf: KDF | None = None,
    drop_locked: bool = False,
) -> OptimizeReport:
    """Rebuild the backing file, dropping every chunk marked DEAD.

    No password is required. ``passwords`` is only consulted when
    ``drop_locked=True`` — in that case, any ``FLAG_OCCUPIED`` slot that
    none of the supplied passwords can unlock is *freed* (its wrap
    cleared and every chunk the slot owner ever wrote marked dead).
    Without ``drop_locked``, locked slots pass through untouched.
    """
    path = Path(path)
    kdf = kdf or KDF()

    if _looks_like_fuse_mount(path):
        raise OptimizeError(f'{path} appears to be mounted; unmount before optimizing')

    old_size = path.stat().st_size

    src_fw = FileWrapper(path)
    src_cover = CoverStorage.attach(src_fw)
    src_container = Container(src_cover)
    cover_length = src_cover.cover_length

    rebuilt_slots: list[int] = sorted(i for i in range(N_SLOTS) if src_container.read_slot(i)[0] == FLAG_OCCUPIED)
    dropped_slots: list[int] = []
    if drop_locked:
        dropped_slots = _identify_locked_slots(src_container, kdf, passwords)
        rebuilt_slots = [i for i in rebuilt_slots if i not in dropped_slots]

    tmp_path = path.with_suffix(path.suffix + '.tmp')
    try:
        _build_compacted(tmp_path, src_container, cover_length, dropped_slots)
        # Release source handles before the atomic replace so no
        # lingering fd holds the pre-rename inode.
        del src_container, src_cover, src_fw
        os.replace(tmp_path, path)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    new_size = path.stat().st_size
    return OptimizeReport(
        old_size=old_size,
        new_size=new_size,
        rebuilt_slots=rebuilt_slots,
        dropped_slots=dropped_slots,
    )


def _identify_locked_slots(src_container: Container, kdf: KDF, passwords: Sequence[str]) -> list[int]:
    """Return occupied slot indices no supplied password can unlock."""
    locked: list[int] = []
    for slot_index in range(N_SLOTS):
        if src_container.read_slot(slot_index)[0] != FLAG_OCCUPIED:
            continue
        if not _any_password_unlocks(src_container, kdf, slot_index, passwords):
            locked.append(slot_index)
    return locked


def _any_password_unlocks(container: Container, kdf: KDF, slot_index: int, passwords: Sequence[str]) -> bool:
    for pw in passwords:
        st = SlotTable(container, kdf, pw)
        if st.is_empty_password and slot_index != 0:
            continue
        if not st.is_empty_password and slot_index == 0:
            continue
        slot_blob = container.read_slot(slot_index)
        if st._unwrap(slot_blob, slot_index) is not None:
            return True
    return False


def _build_compacted(
    tmp_path: Path,
    src: Container,
    cover_length: int,
    dropped_slots: Sequence[int],
) -> None:
    """Build a compacted container at ``tmp_path`` preserving all logical ids."""
    _write_cover_prefix(tmp_path, cover_length, src)

    dst_fw = FileWrapper(tmp_path)
    dst_cover = CoverStorage.attach(dst_fw)
    dst = Container(dst_cover)

    # Copy header (salt only — version and alloc head are re-initialised
    # by the fresh container creation) and slot table.
    dst.write_header(src.read_header())
    slot_table = bytearray(src.read_slot_table())
    for idx in dropped_slots:
        # Zero the slot flag — slot becomes free — and re-randomise the
        # remaining 79 bytes so the freed slot looks indistinguishable
        # from any other free slot.
        slot_table[idx * 80] = FLAG_FREE
        slot_table[idx * 80 + 1 : (idx + 1) * 80] = os.urandom(79)
    dst.write_slot_table(bytes(slot_table))

    # Preserve the reserved flags field verbatim (defence in depth).
    dst_cover.write(FLAGS_OFFSET, src_cover_flags(src))

    # Replay the allocation: every live logical id copies its frame
    # verbatim, every dead logical id is reserved as DEAD.
    src_alloc = src.allocation
    dst_alloc = dst.allocation
    for logical_id in range(src_alloc.next_logical_id):
        physical = src_alloc.lookup(logical_id)
        if physical is None:
            new_id = dst_alloc.append_dead()
        else:
            frame = src_alloc.read(logical_id)
            new_id = dst_alloc.append(frame)
        assert new_id == logical_id, f'logical id drift at {logical_id}: dst got {new_id}'


def src_cover_flags(src: Container) -> bytes:
    return src.storage.read(4, FLAGS_OFFSET)


def _write_cover_prefix(tmp_path: Path, cover_length: int, src: Container) -> None:
    """Copy the source's cover bytes to ``tmp_path`` before container init."""
    if cover_length == 0:
        tmp_path.write_bytes(b'')
        return
    # ``src`` is a Container over a CoverStorage wrapping a FileWrapper.
    # We need the raw file path to copy cover bytes verbatim; narrow
    # via isinstance so pyrefly is happy with the attribute accesses.
    src_storage = src.storage
    assert isinstance(src_storage, CoverStorage)
    inner = src_storage._inner
    assert isinstance(inner, FileWrapper)
    with inner.path.open('rb') as fh, tmp_path.open('wb') as dst:
        remaining = cover_length
        while remaining:
            block = fh.read(min(remaining, 1 << 20))
            if not block:
                break
            dst.write(block)
            remaining -= len(block)
