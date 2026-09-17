"""Read byte-split GGUF files through one contiguous Linux file mapping."""

from __future__ import annotations

import ctypes
import mmap
import os
import re
import sys
from contextlib import ExitStack
from pathlib import Path

import gguf
import numpy as np


def open_gguf_readers(path: str) -> list[gguf.GGUFReader]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"No GGUF checkpoint found at {path}")
    files = sorted(source.glob("*.gguf")) if source.is_dir() else [source]
    parts = sorted(source.glob("*.gguf.part*")) if source.is_dir() else []
    if not source.is_dir() and (
        match := re.fullmatch(r"(.+\.gguf)\.part(\d+)of(\d+)", source.name)
    ):
        files = []
        parts = sorted(source.parent.glob(f"{match[1]}.part*"))
    if parts:
        if files:
            raise ValueError("Both GGUF files and byte-split parts found; select one checkpoint")
        matches = [re.fullmatch(r"(.+\.gguf)\.part(\d+)of(\d+)", part.name) for part in parts]
        if any(match is None for match in matches):
            raise ValueError("Expected GGUF byte-split filenames: NAME.gguf.partNofM")
        names = {match[1] for match in matches}
        counts = {int(match[3]) for match in matches}
        if len(names) != 1 or len(counts) != 1:
            raise ValueError("Pass all byte-split parts of exactly one GGUF checkpoint")
        numbered = sorted((int(match[2]), part) for match, part in zip(matches, parts))
        if counts != {len(parts)} or [n for n, _ in numbered] != list(range(1, len(parts) + 1)):
            raise ValueError("Incomplete or duplicate GGUF parts: expected every part from 1 to M")
        return [_PartsReader([part for _, part in numbered])]
    if not files or any(not file.is_file() or file.suffix != ".gguf" for file in files):
        raise FileNotFoundError(f"No GGUF checkpoint found at {path}")
    return [gguf.GGUFReader(str(file), mode="r") for file in files]


class _MappedParts:
    """Own the mapping; NumPy's base reference keeps it alive for every tensor view."""

    def __init__(self, parts: list[Path]):
        self._address = None
        if sys.platform != "linux" or ctypes.sizeof(ctypes.c_void_p) != 8:
            raise RuntimeError("GGUF .partNofM loading requires 64-bit Linux")
        self._libc = ctypes.CDLL(None, use_errno=True)
        self._libc.mmap.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int64,
        ]
        self._libc.mmap.restype = ctypes.c_void_p
        self._libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self._libc.munmap.restype = ctypes.c_int
        with ExitStack() as stack:
            handles = [stack.enter_context(part.open("rb")) for part in parts]
            sizes = [os.fstat(handle.fileno()).st_size for handle in handles]
            if any(size <= 0 for size in sizes):
                raise ValueError("GGUF parts must not be empty")
            if any(size % mmap.PAGESIZE for size in sizes[:-1]):
                raise ValueError(
                    f"Every GGUF part except the last must have a size divisible by {mmap.PAGESIZE}"
                )
            self._size = sum(sizes)
            self._address = self._map(
                None, self._size, 0, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, -1
            )
            try:
                offset = 0
                for handle, size in zip(handles, sizes):
                    # MAP_FIXED replaces ONLY pages in our reserved PROT_NONE range.
                    self._map(
                        self._address + offset,
                        size,
                        mmap.PROT_READ,
                        mmap.MAP_PRIVATE | 0x10,
                        handle.fileno(),
                    )
                    offset += size
            except BaseException:
                self._unmap()
                raise

    def _map(self, address, size, prot, flags, fd):
        result = self._libc.mmap(address, size, prot, flags, fd, 0)
        if result == ctypes.c_void_p(-1).value:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        return result

    @property
    def __array_interface__(self):
        return {
            "version": 3,
            "shape": (self._size,),
            "typestr": "|u1",
            "data": (self._address, True),
        }

    def _unmap(self):
        if self._address is not None:
            self._libc.munmap(self._address, self._size)
            self._address = None

    def __del__(self):
        self._unmap()


class _PartsReader(gguf.GGUFReader):
    def __init__(self, parts: list[Path]):
        self._part_data = np.asarray(_MappedParts(parts))
        super().__init__(str(parts[0]), mode="r")

    def _get(self, offset, dtype, count=1, override_order=None):
        # GGUFReader initially maps the first file; replace that view on its first read.
        # Reuse the upstream parser, including tensor layout and endianness handling.
        self.data = self._part_data
        if (
            offset < 0
            or count < 0
            or offset + int(count) * np.dtype(dtype).itemsize > self.data.size
        ):
            raise ValueError("Truncated GGUF parts: read extends beyond the complete checkpoint")
        return super()._get(offset, dtype, count, override_order)
