"""정리된 데이터 분할(27,050 / 498 / 360 / 1,198)과 새 Drive zip 처리 테스트"""
import zipfile

from nbcells import load


def _zip(path, name, content=b"x"):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(name, content)


def test_clean_splits_loaded():
    ns = load(["data"])
    assert (len(ns["train_mix"]), len(ns["val_ext"]), len(ns["test_ext"]), len(ns["val_hard"])) == (27050, 498, 360, 1198)
    assert len(ns["train_mix_v1"]) == 12000


def test_no_overlap_between_splits():
    ns = load(["data"])
    tr, ve, te = set(ns["train_mix"].path), set(ns["val_ext"].path), set(ns["test_ext"].path)
    assert not tr & ve and not tr & te and not ve & te
    assert not set(ns["val_hard"].orig_path) & tr


def test_data_zips_include_extra():
    ns = load([])
    assert {"aihub_extra.zip", "extra_public.zip"} <= ns["DATA_ZIPS"]


def test_unpack_reextracts_when_zip_set_changes(tmp_path):
    ns = load([])
    src, root = tmp_path / "src", tmp_path / "root"
    src.mkdir()
    _zip(src / "split_and_results.zip", "a.txt")
    ns["unpack_drive_data"](src, root)
    _zip(src / "extra_public.zip", "b.txt")
    ns["unpack_drive_data"](src, root)
    assert (root / "b.txt").exists()
