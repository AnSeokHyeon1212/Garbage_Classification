"""최종 리뷰 지적 사항 재현 테스트 (#1 가중치 없는 Drive, #2 데이터 누락·모르는 zip, #3 설정 불일치 재사용, #4 TEST 폴더)"""
import zipfile

import pandas as pd
import pytest

from nbcells import load

TAGS = ["data", "cache", "model", "train", "ladder"]


def _zip(path, name, content=b"x"):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(name, content)


def test_unpack_without_weights(tmp_path):
    """#1: Drive에 final_model.pth가 없어도 압축 해제는 끝나야 한다"""
    ns = load([])
    src = tmp_path / "src"
    (src / "05_val_hard_generated").mkdir(parents=True)
    _zip(src / "05_val_hard_generated" / "val_hard_generated.zip", "external_data/val_hard/a.txt")
    ns["unpack_drive_data"](src, tmp_path / "root")
    assert (tmp_path / "root" / ".prepared").exists()
    assert (tmp_path / "root" / "external_data/val_hard/a.txt").exists()


def test_unpack_only_known_zips(tmp_path):
    """#2: 알려진 데이터 zip만 푼다 (예: AI Hub 원본 대용량 zip이 섞여 있어도 무시)"""
    ns = load([])
    src = tmp_path / "src"
    src.mkdir()
    _zip(src / "split_and_results.zip", "01_coding/RESULT/data_split/x.csv")
    _zip(src / "TS_2.직접촬영_원본.zip", "huge/raw.jpg")
    ns["unpack_drive_data"](src, tmp_path / "root")
    assert (tmp_path / "root/01_coding/RESULT/data_split/x.csv").exists()
    assert not (tmp_path / "root/huge").exists()


def test_full_data_guard_blocks_missing_images():
    """#2: 실험 실행 전, 검증·학습 이미지가 빠져 있으면 멈춘다"""
    ns = load(TAGS)
    bad = pd.concat([ns["val_ext"].head(3), pd.DataFrame([{"path": "no/such.jpg", "label": "metal", "source": "aihub"}])])
    with pytest.raises(RuntimeError, match="이미지가 없음"):
        ns["assert_full_data"]({"val_ext": bad})
    ns["assert_full_data"]({"val_ext": ns["val_ext"].head(3)})   # 다 있으면 통과


def test_resume_rejects_changed_cfg(tmp_path):
    """#3: 같은 ID의 저장 결과가 다른 설정으로 만들어졌으면 재사용하지 않고 오류"""
    ns = load(TAGS, CACHE_ROOT=tmp_path / "c", SUBSET_FOR_TEST=16)
    cfg = {**ns["BASE_CFG"], "epochs": 1, "batch": 8, "img_size": 64}
    ns["run_experiment"]("T1", cfg, tmp_path / "exp")
    ns["run_experiment"]("T1", dict(cfg), tmp_path / "exp")          # 같은 설정 → 재사용
    with pytest.raises(ValueError, match="설정이 다름"):
        ns["run_experiment"]("T1", {**cfg, "lr": 3e-3}, tmp_path / "exp")


def test_final_test_dir_local():
    """#4: 로컬은 RESULT/final_test, Colab은 Drive 폴더(있을 때)"""
    ns = load([])
    assert ns["final_test_dir"](False) == ns["ROOT"] / "01_coding" / "RESULT" / "final_test"
