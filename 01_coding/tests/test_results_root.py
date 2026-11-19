"""다른 계정의 Colab에서 실행할 때: 공개 폴더에서 데이터를 받고, 결과는 그 계정의 내 Drive에 남긴다"""
from nbcells import load


def test_results_root_owner(tmp_path):
    ns = load([])
    (tmp_path / ns["DRIVE_NAME"]).mkdir()
    assert ns["results_root"](True, tmp_path) == tmp_path / ns["DRIVE_NAME"]


def test_results_root_other_account(tmp_path):
    ns = load([])
    assert ns["results_root"](True, tmp_path) == tmp_path / ns["RUNS_NAME"]


def test_results_root_no_drive_or_local(tmp_path):
    ns = load([])
    local = ns["ROOT"] / "01_coding" / "RESULT"
    assert ns["results_root"](True, tmp_path / "none") == local
    assert ns["results_root"](False, tmp_path) == local


def test_seed_results_copies_missing_only(tmp_path):
    ns = load([])
    src, dst = tmp_path / "src", tmp_path / "dst"
    (src / "experiments" / "R1").mkdir(parents=True)
    (src / "experiments" / "ladder.csv").write_text("old")
    (src / "experiments" / "R1" / "metrics.json").write_text("{}")
    (src / "final_test").mkdir()
    (src / "final_test" / "final_test.json").write_text("{}")
    (dst / "experiments").mkdir(parents=True)
    (dst / "experiments" / "ladder.csv").write_text("mine")
    ns["seed_results"](src, dst)
    assert (dst / "experiments" / "ladder.csv").read_text() == "mine"       # 이미 있으면 덮어쓰지 않음
    assert (dst / "experiments" / "R1" / "metrics.json").exists()
    assert (dst / "final_test" / "final_test.json").exists()               # TEST 1회 기록도 함께 옮김


def test_seed_results_same_folder_noop(tmp_path):
    ns = load([])
    (tmp_path / "experiments").mkdir()
    ns["seed_results"](tmp_path, tmp_path)   # 같은 폴더면 아무것도 안 함 (오류 없음)
