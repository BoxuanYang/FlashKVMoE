"""Real Linux mappings: cross-file metadata/tensors, lifetime, and invalid parts."""

import gc
import mmap
import sys
import weakref
from pathlib import Path

import gguf
import numpy as np
import pytest
import torch
from minisgl.models.gguf_parts import _MappedParts, open_gguf_readers

linux = pytest.mark.skipif(sys.platform != "linux", reason="Contiguous file mapping needs Linux")


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "model.gguf"
    writer = gguf.GGUFWriter(str(path), "deepseek2")
    writer.add_string("test.padding", "x" * 8192)
    writer.add_tensor("dense", np.arange(4096, dtype=np.float32))
    packed = gguf.quantize(
        np.arange(32768, dtype=np.float32).reshape(128, 256), gguf.GGMLQuantizationType.Q8_0
    )
    writer.add_tensor("experts", packed, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    content = path.read_bytes()
    directory = tmp_path / "parts"
    directory.mkdir()
    count = (len(content) + 4095) // 4096
    assert count > 10  # Exercise numeric order: part10 must not precede part2.
    parts = []
    for i in range(count):
        part = directory / f"model.gguf.part{i + 1}of{count}"
        part.write_bytes(content[i * 4096 : (i + 1) * 4096])
        parts.append(part)
    return path, directory, parts


@linux
@pytest.mark.parametrize("entry", ["directory", "part"])
def test_parts_match_unsplit_without_copy(checkpoint, entry):
    path, directory, parts = checkpoint
    before = set(directory.iterdir())
    reference = gguf.GGUFReader(str(path))
    (reader,) = open_gguf_readers(str(directory if entry == "directory" else parts[-1]))
    assert reader.data.tobytes() == path.read_bytes()
    # The payload is backed by the original files, not an anonymous concatenation.
    mapped_paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        address, permissions, _, _, _, *filename = line.split(maxsplit=5)
        start, end = (int(x, 16) for x in address.split("-"))
        if reader.data.ctypes.data <= start < reader.data.ctypes.data + reader.data.size:
            assert permissions == "r--p" and end > start
            mapped_paths.add(filename[0])
    assert mapped_paths == {str(part.resolve()) for part in parts}
    assert reader.fields.keys() == reference.fields.keys()
    for name in reader.fields:
        assert reader.fields[name].contents() == reference.fields[name].contents()
    for actual, expected in zip(reader.tensors, reference.tensors):
        assert actual.name == expected.name
        assert actual.tensor_type == expected.tensor_type
        np.testing.assert_array_equal(actual.shape, expected.shape)
        np.testing.assert_array_equal(actual.data, expected.data)
        assert np.shares_memory(actual.data, reader.data)
        assert not actual.data.flags.writeable
        assert actual.data.ctypes.data == reader.data.ctypes.data + actual.data_offset
        assert actual.data.nbytes > mmap.PAGESIZE
        np.testing.assert_array_equal(
            gguf.dequantize(actual.data, actual.tensor_type),
            gguf.dequantize(expected.data, expected.tensor_type),
        )
    assert set(directory.iterdir()) == before


@linux
@pytest.mark.parametrize("torch_view", [False, True])
def test_views_keep_mapping_alive(checkpoint, torch_view):
    _, directory, _ = checkpoint
    (reader,) = open_gguf_readers(str(directory))
    owner = weakref.ref(reader.data.base)
    view = reader.tensors[0].data.reshape(-1)[-16:]
    expected = view.copy()
    if torch_view:
        view = torch.from_numpy(view)
    del reader
    gc.collect()
    assert owner() is not None
    np.testing.assert_array_equal(view, expected)
    del view
    gc.collect()
    assert owner() is None


@pytest.mark.parametrize("problem", ["missing", "duplicate", "mixed_count", "mixed_name", "merged"])
def test_invalid_part_sets(checkpoint, problem):
    path, directory, parts = checkpoint
    if problem == "missing":
        parts[1].unlink()
    elif problem == "duplicate":
        (directory / parts[0].name.replace("part1of", "part01of")).write_bytes(
            parts[0].read_bytes()
        )
    elif problem == "mixed_count":
        parts[-1].rename(directory / f"model.gguf.part{len(parts)}of99")
    elif problem == "mixed_name":
        parts[-1].rename(directory / f"other.gguf.part{len(parts)}of{len(parts)}")
    else:
        (directory / "model.gguf").write_bytes(path.read_bytes())
    with pytest.raises(ValueError):
        open_gguf_readers(str(directory))


@linux
@pytest.mark.parametrize(
    "problem,match",
    [
        ("empty", "must not be empty"),
        ("unaligned", "divisible"),
        ("truncated", "Truncated GGUF"),
    ],
)
def test_invalid_part_sizes(checkpoint, problem, match):
    _, directory, parts = checkpoint
    part = parts[-1] if problem == "truncated" else parts[0]
    part.write_bytes(b"" if problem == "empty" else part.read_bytes()[:-1])
    with pytest.raises(ValueError, match=match):
        open_gguf_readers(str(directory))


@linux
def test_large_sparse_parts_use_64_bit_offsets(tmp_path):
    first, last = tmp_path / "large.part1", tmp_path / "large.part2"
    size = 45097156608  # Exercise offsets beyond 32 bits without allocating disk blocks.
    with first.open("wb") as file:
        file.truncate(size)
        file.seek(size - 4)
        file.write(b"abcd")
    last.write_bytes(b"efghijk")
    data = np.asarray(_MappedParts([first, last]))
    assert data.size == size + 7
    assert data[size - 4 :].tobytes() == b"abcdefghijk"
    assert not data.flags.writeable


@linux
def test_failed_mapping_releases_reservation(checkpoint, monkeypatch):
    _, directory, _ = checkpoint
    original = _MappedParts._map
    owners = []

    def fail_on_file(self, address, size, prot, flags, fd):
        if address is not None:
            owners.append(self)
            if len(owners) == 2:
                raise OSError("test mapping failure")
        return original(self, address, size, prot, flags, fd)

    monkeypatch.setattr(_MappedParts, "_map", fail_on_file)
    with pytest.raises(OSError, match="test mapping failure"):
        open_gguf_readers(str(directory))
    assert len(owners) == 2 and owners[0]._address is None
