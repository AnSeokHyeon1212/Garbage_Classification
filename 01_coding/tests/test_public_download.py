"""공개 Drive 폴더 내려받기: 큰 파일 확인 페이지 우회, 필요한 파일만, 재시도, 이미 받은 파일 건너뛰기"""
from collections import namedtuple

import pytest

from nbcells import load

F = namedtuple("F", "id path local_path")


def _files(out):
    return [F("a", "01_x/kaggle_garbage_classification.zip", str(out / "01_x/kaggle_garbage_classification.zip")),
            F("b", "experiments/R1/metrics.json", str(out / "experiments/R1/metrics.json")),
            F("c", "04_y/TS_원본_대용량.zip", str(out / "04_y/TS_원본_대용량.zip")),     # 모르는 zip → 받지 않음
            F("d", "README.txt", str(out / "README.txt"))]


def test_downloads_only_needed_and_skips_existing(tmp_path):
    ns = load([])
    (tmp_path / "experiments/R1").mkdir(parents=True)
    (tmp_path / "experiments/R1/metrics.json").write_text("{}")      # 이미 있음 → 건너뜀
    got = []

    def fake(fid, dst):
        got.append(fid)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"x")
        return True

    ns["download_public_folder"]("FOLDER", tmp_path, list_files=lambda: _files(tmp_path), download=fake, wait=0)
    assert got == ["a"]


def test_retries_then_succeeds(tmp_path):
    ns = load([])
    calls = []

    def flaky(fid, dst):
        calls.append(fid)
        if len(calls) < 3:
            return False
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"x")
        return True

    ns["download_public_folder"]("FOLDER", tmp_path, list_files=lambda: _files(tmp_path)[:1], download=flaky, wait=0)
    assert len(calls) == 3 and (tmp_path / "01_x/kaggle_garbage_classification.zip").exists()


def test_gives_clear_error_after_retries(tmp_path):
    ns = load([])
    with pytest.raises(RuntimeError, match="바로가기"):
        ns["download_public_folder"]("FOLDER", tmp_path, list_files=lambda: _files(tmp_path)[:1],
                                     download=lambda fid, dst: False, wait=0)


class _Resp:
    def __init__(self, ctype, body=b"data"):
        self.status_code, self.headers, self._b = 200, {"Content-Type": ctype}, body

    def iter_content(self, n):
        yield self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Sess:
    def __init__(self, resp):
        self.resp, self.params = resp, None

    def get(self, url, params=None, **k):
        self.params = params
        return self.resp


def test_direct_download_writes_file_and_rejects_html(tmp_path):
    ns = load([])
    s = _Sess(_Resp("application/zip"))
    assert ns["drive_file_download"]("ID", tmp_path / "a.zip", session=s) is True
    assert (tmp_path / "a.zip").read_bytes() == b"data" and s.params["confirm"] == "t"
    assert ns["drive_file_download"]("ID", tmp_path / "b.zip", session=_Sess(_Resp("text/html; charset=utf-8"))) is False
    assert not (tmp_path / "b.zip").exists()
