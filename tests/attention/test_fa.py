import importlib.util
from pathlib import Path
from types import SimpleNamespace

MODULE_PATH = Path(__file__).parents[2] / "python/minisgl/attention/_fa_import.py"
SPEC = importlib.util.spec_from_file_location("minisgl_fa_import", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
fa_import = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fa_import)


def test_fa3_import_skips_optional_fa4(monkeypatch):
    def import_module(name):
        assert name == "sgl_kernel.flash_attn"
        assert fa_import.sys.modules["sgl_kernel._fa4_interface"].flash_attn_varlen_func is None
        return SimpleNamespace(flash_attn_with_kvcache="fa3")

    monkeypatch.delitem(fa_import.sys.modules, "sgl_kernel._fa4_interface", raising=False)
    monkeypatch.setattr(fa_import.importlib, "import_module", import_module)

    assert fa_import.load_flash_attn_with_kvcache(3) == "fa3"
    assert "sgl_kernel._fa4_interface" not in fa_import.sys.modules


def test_fa4_import_is_not_stubbed(monkeypatch):
    def import_module(name):
        assert name == "sgl_kernel.flash_attn"
        assert "sgl_kernel._fa4_interface" not in fa_import.sys.modules
        return SimpleNamespace(flash_attn_with_kvcache="fa4")

    monkeypatch.delitem(fa_import.sys.modules, "sgl_kernel._fa4_interface", raising=False)
    monkeypatch.setattr(fa_import.importlib, "import_module", import_module)

    assert fa_import.load_flash_attn_with_kvcache(4) == "fa4"
