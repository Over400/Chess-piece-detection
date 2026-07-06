"""
체스 기물 Detection (YOLOv8) — 로봇팔 체스용 1단계 모델

  Kaggle 데이터 → YOLO 형식 → train → best.pt 추론

우선 데이터셋:
  imtkaggleteam/chess-pieces-detection-image-dataset
선택 추가:
  tannergi/chess-piece-detection

사용법:
    cd ~/디렉터리 경로
    python3 model_alpha_v2.py prepare --source huggingface
    python3 model_alpha_v2.py train --epochs 50
    python3 model_alpha_v2.py predict --source 이미지경로.jpg

설명:
  python model_alpha_v2.py prepare                      # 다운로드 + YOLO 정리, 데이터 준비
  python model_alpha_v2.py prepare --extra              # tannergi 도 합치기
  python model_alpha_v2.py train                        # 학습, best.pt 저장
  python model_alpha_v2.py train --epochs 30 --batch 8  # 학습 설정 변경
  python model_alpha_v2.py predict --source board.jpg   # 추론, 결과저장
  python model_alpha_v2.py all                          # prepare → train → 샘플 추론

--- 타 PC 듀얼 웹캠 로봇 체스 (chess_dual_cam_pipeline.py) ---

  [fixed cam]  거치대 — 보드 전체 → 국면 모니터링·변화 감지
  [gripper cam] 그리퍼 근처 — 집기/놓기 국소 검증

  MONITOR → CHANGED → DECIDE → EXECUTE → VERIFY → MONITOR
  본 파일(best.pt) = 기물 detector / 파이프라인 = 카메라·변화·로봇 연동
"""

from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# 경로 / 데이터셋
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
WORK_DIR = ROOT / "chess_workspace"
RAW_DIR = WORK_DIR / "kaggle_raw"
YOLO_DIR = WORK_DIR / "chess_yolov8"
RUNS_NAME = "chess_pieces"
PRETRAINED = ROOT / "yolov8n.pt"

PRIMARY_DATASET = "imtkaggleteam/chess-pieces-detection-image-dataset"
EXTRA_DATASET = "tannergi/chess-piece-detection"
HF_DATASET = "acapitani/chesspiece-detection-yolo"
HF_ARCHIVE = "dataset.tar.gz"

CHESS_CLASS_NAMES = [
    "white_pawn",
    "white_rook",
    "white_knight",
    "white_bishop",
    "white_queen",
    "white_king",
    "black_pawn",
    "black_rook",
    "black_knight",
    "black_bishop",
    "black_queen",
    "black_king",
    "empty_square",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
    "testing": "test",
}


def device_str() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def best_weights_path() -> Path:
    return ROOT / "runs" / "detect" / RUNS_NAME / "weights" / "best.pt"


# ---------------------------------------------------------------------------
# Kaggle 다운로드
# ---------------------------------------------------------------------------

def kaggle_credentials() -> Path | None:
    for cred in (
        Path.home() / ".kaggle" / "kaggle.json",
        Path.home() / ".config" / "kaggle" / "kaggle.json",
    ):
        if cred.exists():
            return cred
    return None


def ensure_kaggle() -> None:
    cred = kaggle_credentials()
    if cred is None:
        raise FileNotFoundError(
            "Kaggle API 인증이 필요합니다.\n"
            "1) https://www.kaggle.com/settings → Create New Token\n"
            "2) kaggle.json 을 ~/.kaggle/kaggle.json 에 저장\n"
            "3) chmod 600 ~/.kaggle/kaggle.json"
        )
    try:
        import kaggle  # noqa: F401
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "kaggle"]
        )


def download_dataset(slug: str, dest: Path) -> Path:
    """Kaggle 데이터셋을 dest/<slug_name> 에 다운로드·압축 해제."""
    ensure_kaggle()
    dest.mkdir(parents=True, exist_ok=True)
    name = slug.split("/")[-1]
    out = dest / name
    if out.exists() and any(out.rglob("*")):
        print(f"[skip] 이미 있음: {out}")
        return out

    print(f"[download] {slug} → {out}")
    subprocess.check_call(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            slug,
            "--unzip",
            "-p",
            str(out),
        ]
    )
    return out


def download_huggingface_dataset(dest: Path) -> Path:
    """Kaggle 없을 때 Hugging Face 공개 YOLO 체스 데이터셋 사용."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"]
        )
        from huggingface_hub import hf_hub_download

    dest.mkdir(parents=True, exist_ok=True)
    dataset_root = dest / "extracted" / "dataset"
    if (dataset_root / "images" / "train").exists():
        print(f"[skip] HuggingFace 데이터셋 이미 있음: {dataset_root}")
        return dataset_root

    print(f"[download] HuggingFace {HF_DATASET}")
    archive = hf_hub_download(
        repo_id=HF_DATASET,
        filename=HF_ARCHIVE,
        repo_type="dataset",
        local_dir=str(dest),
    )
    import tarfile

    print(f"[extract] {archive}")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(dest / "extracted")
    return dataset_root


def write_data_yaml(dataset_root: Path, class_names: list[str]) -> Path:
    """기존 YOLO 폴더를 가리키는 data.yaml 생성."""
    YOLO_DIR.mkdir(parents=True, exist_ok=True)
    yaml_path = YOLO_DIR / "data.yaml"

    train = "images/train"
    val = "images/val"
    test = "images/test"
    if not (dataset_root / "images" / "val").exists():
        val = train
    if not (dataset_root / "images" / "test").exists():
        test = val

    data_cfg = {
        "path": str(dataset_root.resolve()),
        "train": train,
        "val": val,
        "test": test,
        "nc": len(class_names),
        "names": class_names,
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print("\n=== YOLO data.yaml 준비 완료 ===")
    print(f"dataset: {dataset_root}")
    print(f"config : {yaml_path}")
    for split in ("train", "val", "test"):
        split_dir = dataset_root / "images" / split
        if split_dir.exists():
            n_img = len([p for p in split_dir.iterdir() if _is_image(p)])
            print(f"  {split}: {n_img} images")
    print("클래스:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")
    return yaml_path


def normalize_class_names(names: list[str] | None, nc: int) -> list[str]:
    """
    원본 데이터셋에 의미있는 클래스 이름이 없을 때(숫자 id만 있거나,
    아예 없어서 'class_0' 식으로 생성된 경우) CHESS_CLASS_NAMES 순서를
    "추정값"으로 사용한다.

    주의: 이건 어디까지나 추정이다. 원본 Kaggle/HF 데이터셋이 실제로
    이 순서(white_pawn=0 ... empty_square=12)를 따른다는 보장이 없으므로,
    학습 전에 반드시 원본 데이터셋 페이지의 클래스 정의와 직접 대조해야 한다.
    잘못 추정하면 mAP/loss에는 안 잡히고 로봇이 엉뚱한 기물로 착각하는
    조용한(silent) 오류로 이어진다.
    """
    is_placeholder = not names or all(
        str(n).isdigit() or str(n).startswith("class_") for n in names
    )
    if is_placeholder:
        guessed = CHESS_CLASS_NAMES[:nc]
        print(
            "\n[!!! 클래스 이름 추정 경고 !!!]\n"
            f"  원본에서 읽은 이름: {names!r}\n"
            f"  추정해서 사용할 이름 (순서 가정): {guessed}\n"
            "  -> 이 추정이 실제 데이터셋 라벨 순서와 다르면 잘못된 클래스가 학습됩니다.\n"
            "  -> 반드시 Kaggle/HF 데이터셋 설명 페이지에서 class id 순서를 확인하세요.\n"
        )
        return guessed
    return names


# ---------------------------------------------------------------------------
# 데이터셋 형식 탐색 / YOLO 정규화
# ---------------------------------------------------------------------------

def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def find_data_yaml(root: Path) -> Path | None:
    for name in ("data.yaml", "dataset.yaml", "data.yml"):
        hits = list(root.rglob(name))
        if hits:
            return hits[0]
    return None


def load_names_from_yaml(yaml_path: Path) -> list[str] | None:
    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    names = cfg.get("names")
    if names is None:
        return None
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=lambda x: int(x))]
    return list(names)


def collect_split_pairs(root: Path) -> dict[str, list[tuple[Path, Path]]]:
    """
    여러 YOLO 레이아웃에서 (image, label) 쌍을 split별로 수집.

    지원:
      - images/{split}/*.jpg + labels/{split}/*.txt
      - {split}/images/*.jpg + {split}/labels/*.txt
      - {split}/*.jpg + {split}/*.txt (같은 폴더)
    """
    pairs: dict[str, list[tuple[Path, Path]]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    # A) images/split + labels/split
    images_root = None
    for candidate in root.rglob("images"):
        if candidate.is_dir():
            images_root = candidate
            break
    if images_root is not None:
        labels_root = images_root.parent / "labels"
        if labels_root.is_dir():
            for split_dir in images_root.iterdir():
                if not split_dir.is_dir():
                    continue
                split = SPLIT_ALIASES.get(split_dir.name.lower())
                if not split:
                    continue
                label_dir = labels_root / split_dir.name
                if not label_dir.is_dir():
                    # valid vs val 등
                    for alt in ("val", "valid", "validation"):
                        if (labels_root / alt).is_dir() and split == "val":
                            label_dir = labels_root / alt
                            break
                for img in split_dir.iterdir():
                    if not _is_image(img):
                        continue
                    lbl = label_dir / f"{img.stem}.txt"
                    if lbl.exists():
                        pairs[split].append((img, lbl))

    # B) split/images + split/labels
    for split_name, split_key in SPLIT_ALIASES.items():
        for split_dir in root.rglob(split_name):
            if not split_dir.is_dir():
                continue
            img_dir = split_dir / "images"
            lbl_dir = split_dir / "labels"
            if not (img_dir.is_dir() and lbl_dir.is_dir()):
                continue
            for img in img_dir.iterdir():
                if not _is_image(img):
                    continue
                lbl = lbl_dir / f"{img.stem}.txt"
                if lbl.exists():
                    pairs[split_key].append((img, lbl))

    # 중복 제거
    for split in pairs:
        seen: set[str] = set()
        unique: list[tuple[Path, Path]] = []
        for img, lbl in pairs[split]:
            key = img.name
            if key in seen:
                continue
            seen.add(key)
            unique.append((img, lbl))
        pairs[split] = unique

    return pairs


def find_voc_dirs(root: Path) -> tuple[Path, Path] | None:
    """Pascal VOC: images/ + annotations/*.xml"""
    xmls = list(root.rglob("*.xml"))
    if not xmls:
        return None
    ann_dir = xmls[0].parent
    # 이미지 폴더 추정
    for name in ("images", "JPEGImages", "img", "imgs"):
        for d in root.rglob(name):
            if d.is_dir() and any(_is_image(p) for p in d.iterdir()):
                return d, ann_dir
    # xml과 같은 폴더에 이미지가 있는 경우
    if any(_is_image(p) for p in ann_dir.iterdir()):
        return ann_dir, ann_dir
    return None


def xml_to_yolo_lines(
    xml_file: Path,
    class_to_id: dict[str, int],
) -> list[str]:
    tree = ET.parse(xml_file)
    root = tree.getroot()
    size = root.find("size")
    if size is None:
        return []
    w = float(size.findtext("width", "0"))
    h = float(size.findtext("height", "0"))
    if w <= 0 or h <= 0:
        return []

    lines: list[str] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name not in class_to_id:
            class_to_id[name] = len(class_to_id)
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        xmin = float(bbox.findtext("xmin", "0"))
        ymin = float(bbox.findtext("ymin", "0"))
        xmax = float(bbox.findtext("xmax", "0"))
        ymax = float(bbox.findtext("ymax", "0"))
        xc = ((xmin + xmax) / 2.0) / w
        yc = ((ymin + ymax) / 2.0) / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        cls_id = class_to_id[name]
        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return lines


def convert_voc_tree(raw_root: Path) -> dict[str, list[tuple[Path, Path]]]:
    """VOC → 임시 YOLO 라벨을 raw 옆에 만들고 pair 반환."""
    found = find_voc_dirs(raw_root)
    if found is None:
        return {"train": [], "val": [], "test": []}

    img_dir, ann_dir = found
    class_to_id: dict[str, int] = {}
    tmp_labels = WORK_DIR / "_voc_labels" / raw_root.name
    if tmp_labels.exists():
        shutil.rmtree(tmp_labels)
    tmp_labels.mkdir(parents=True)

    pairs: list[tuple[Path, Path]] = []
    for xml_file in ann_dir.glob("*.xml"):
        img = None
        for ext in IMAGE_EXTS:
            candidate = img_dir / f"{xml_file.stem}{ext}"
            if candidate.exists():
                img = candidate
                break
        if img is None:
            continue
        lines = xml_to_yolo_lines(xml_file, class_to_id)
        lbl = tmp_labels / f"{xml_file.stem}.txt"
        lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        pairs.append((img, lbl))

    # 클래스 순서 저장
    names = [n for n, _ in sorted(class_to_id.items(), key=lambda x: x[1])]
    (tmp_labels / "_names.txt").write_text("\n".join(names), encoding="utf-8")

    random.seed(42)
    random.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * 0.7)
    n_val = int(n * 0.2)
    return {
        "train": pairs[:n_train],
        "val": pairs[n_train : n_train + n_val],
        "test": pairs[n_train + n_val :],
    }


def materialize_yolo_dataset(
    all_pairs: dict[str, list[tuple[Path, Path]]],
    class_names: list[str],
) -> Path:
    """표준 images/{split}, labels/{split} + data.yaml 생성."""
    if YOLO_DIR.exists():
        shutil.rmtree(YOLO_DIR)
    for split in ("train", "val", "test"):
        (YOLO_DIR / "images" / split).mkdir(parents=True)
        (YOLO_DIR / "labels" / split).mkdir(parents=True)

    for split, pairs in all_pairs.items():
        for img, lbl in pairs:
            # 이름 충돌 방지
            dst_img = YOLO_DIR / "images" / split / img.name
            if dst_img.exists():
                dst_img = YOLO_DIR / "images" / split / f"{img.stem}_{img.parent.name}{img.suffix}"
            dst_lbl = YOLO_DIR / "labels" / split / f"{dst_img.stem}.txt"
            shutil.copy2(img, dst_img)
            shutil.copy2(lbl, dst_lbl)

    # val이 비면 train에서 일부 사용
    if not all_pairs.get("val") and all_pairs.get("train"):
        train_imgs = list((YOLO_DIR / "images" / "train").glob("*"))
        move_n = max(1, len(train_imgs) // 5)
        for img in train_imgs[:move_n]:
            lbl = YOLO_DIR / "labels" / "train" / f"{img.stem}.txt"
            shutil.move(str(img), YOLO_DIR / "images" / "val" / img.name)
            if lbl.exists():
                shutil.move(str(lbl), YOLO_DIR / "labels" / "val" / lbl.name)

    data_cfg = {
        "path": str(YOLO_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": class_names,
    }
    yaml_path = YOLO_DIR / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print("\n=== YOLO 데이터셋 준비 완료 ===")
    print(f"경로: {YOLO_DIR}")
    for split in ("train", "val", "test"):
        n_img = len(list((YOLO_DIR / "images" / split).glob("*")))
        n_lbl = len(list((YOLO_DIR / "labels" / split).glob("*.txt")))
        print(f"  {split}: {n_img} images, {n_lbl} labels")
    print("클래스:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")
    return yaml_path


def gather_from_raw(raw_root: Path) -> tuple[dict[str, list[tuple[Path, Path]]], list[str]]:
    """단일 raw 폴더에서 pairs + class names 추출."""
    yaml_path = find_data_yaml(raw_root)
    class_names: list[str] | None = None
    if yaml_path is not None:
        class_names = load_names_from_yaml(yaml_path)
        print(f"[info] data.yaml: {yaml_path}")

    pairs = collect_split_pairs(raw_root)
    total = sum(len(v) for v in pairs.values())
    if total > 0:
        if not class_names:
            # 라벨 id 최대값으로 클래스 수 추정
            max_id = -1
            for split_pairs in pairs.values():
                for _, lbl in split_pairs:
                    for line in lbl.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        max_id = max(max_id, int(line.split()[0]))
            class_names = [f"class_{i}" for i in range(max_id + 1)]
        return pairs, class_names

    # VOC fallback
    print("[info] YOLO 쌍 없음 → VOC XML 변환 시도")
    pairs = convert_voc_tree(raw_root)
    names_file = WORK_DIR / "_voc_labels" / raw_root.name / "_names.txt"
    if names_file.exists():
        class_names = names_file.read_text(encoding="utf-8").splitlines()
    else:
        class_names = class_names or []
    return pairs, class_names


def merge_pairs(
    base: dict[str, list[tuple[Path, Path]]],
    extra: dict[str, list[tuple[Path, Path]]],
) -> dict[str, list[tuple[Path, Path]]]:
    out = {k: list(v) for k, v in base.items()}
    for split, pairs in extra.items():
        out.setdefault(split, []).extend(pairs)
    return out


# ---------------------------------------------------------------------------
# prepare / train / predict
# ---------------------------------------------------------------------------

def cmd_prepare(extra: bool = False, source: str = "auto") -> Path:
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    if source in ("auto", "kaggle"):
        try:
            primary_raw = download_dataset(PRIMARY_DATASET, RAW_DIR)
        except Exception as exc:
            if source == "kaggle":
                raise
            print(f"[warn] Kaggle 사용 불가 → HuggingFace fallback ({exc})")
        else:
            return _prepare_from_kaggle(extra=extra, primary_raw=primary_raw)

    hf_root = download_huggingface_dataset(WORK_DIR / "hf_acapitani")
    yaml_path = find_data_yaml(hf_root)
    class_names = normalize_class_names(
        load_names_from_yaml(yaml_path) if yaml_path else None,
        nc=13,
    )
    return write_data_yaml(hf_root, class_names)


def _prepare_from_kaggle(extra: bool, primary_raw: Path) -> Path:
    pairs, class_names = gather_from_raw(primary_raw)

    if extra:
        try:
            extra_raw = download_dataset(EXTRA_DATASET, RAW_DIR)
            extra_pairs, extra_names = gather_from_raw(extra_raw)
            if extra_names and class_names and extra_names != class_names:
                print(
                    "[warn] 클래스 이름이 다릅니다. primary 클래스 순서를 유지합니다.\n"
                    f"  primary: {class_names}\n"
                    f"  extra:   {extra_names}"
                )
                # 이름이 같으면 id 재매핑, 다르면 extra 스킵
                if set(extra_names) != set(class_names):
                    print("[warn] 클래스 집합이 달라 extra 데이터셋을 합치지 않습니다.")
                else:
                    name_to_id = {n: i for i, n in enumerate(class_names)}
                    remapped: dict[str, list[tuple[Path, Path]]] = {
                        "train": [],
                        "val": [],
                        "test": [],
                    }
                    remap_dir = WORK_DIR / "_remap_labels"
                    if remap_dir.exists():
                        shutil.rmtree(remap_dir)
                    remap_dir.mkdir(parents=True)
                    old_to_new = {
                        i: name_to_id[n] for i, n in enumerate(extra_names)
                    }
                    for split, split_pairs in extra_pairs.items():
                        for img, lbl in split_pairs:
                            lines_out = []
                            for line in lbl.read_text(encoding="utf-8").splitlines():
                                parts = line.strip().split()
                                if len(parts) < 5:
                                    continue
                                old_id = int(parts[0])
                                parts[0] = str(old_to_new[old_id])
                                lines_out.append(" ".join(parts))
                            new_lbl = remap_dir / f"{img.stem}_{split}.txt"
                            new_lbl.write_text(
                                "\n".join(lines_out) + ("\n" if lines_out else ""),
                                encoding="utf-8",
                            )
                            remapped[split].append((img, new_lbl))
                    pairs = merge_pairs(pairs, remapped)
            else:
                pairs = merge_pairs(pairs, extra_pairs)
                if not class_names:
                    class_names = extra_names
        except Exception as e:
            print(f"[warn] extra 데이터셋 스킵: {e}")

    total = sum(len(v) for v in pairs.values())
    if total == 0:
        raise RuntimeError(
            f"라벨된 이미지를 찾지 못했습니다: {primary_raw}\n"
            "폴더 구조를 확인하세요."
        )
    if not class_names:
        raise RuntimeError("클래스 이름을 알 수 없습니다.")

    class_names = normalize_class_names(class_names, len(class_names))
    return materialize_yolo_dataset(pairs, class_names)


def cmd_train(
    epochs: int = 50,
    imgsz: int = 640,
    batch: int | None = None,
    patience: int = 10,
) -> Path:
    yaml_path = YOLO_DIR / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"{yaml_path} 없음. 먼저 실행: python model_alpha.py prepare"
        )

    if batch is None:
        batch = 16 if torch.cuda.is_available() else 4

    weights = str(PRETRAINED) if PRETRAINED.exists() else "yolov8n.pt"
    print(f"[train] device={device_str()}, batch={batch}, epochs={epochs}")
    print(f"[train] data={yaml_path}")

    model = YOLO(weights)
    model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=RUNS_NAME,
        project=str(ROOT / "runs" / "detect"),
        patience=patience,
        save=True,
        device=device_str(),
        exist_ok=True,
        verbose=True,
    )

    best = best_weights_path()
    print(f"[train] best weights: {best}")
    return best


def parse_detections(result, conf_threshold: float = 0.25) -> list[dict[str, Any]]:
    """로봇팔용: 클래스 / 신뢰도 / 박스 / 중심점."""
    names = result.names
    detections: list[dict[str, Any]] = []
    if result.boxes is None:
        return detections

    for box in result.boxes:
        conf = float(box.conf[0].cpu().item())
        if conf < conf_threshold:
            continue
        cls_id = int(box.cls[0].cpu().item())
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
        detections.append(
            {
                "class_id": cls_id,
                "class_name": names[cls_id],
                "confidence": conf,
                "xyxy": [x1, y1, x2, y2],
                "center": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
            }
        )
    return detections


def cmd_predict(
    source: str | Path,
    weights: Path | None = None,
    conf: float = 0.25,
    save: bool = True,
) -> list[dict[str, Any]]:
    weights = weights or best_weights_path()
    if not weights.exists():
        raise FileNotFoundError(
            f"가중치 없음: {weights}\n먼저 학습: python model_alpha.py train"
        )

    model = YOLO(str(weights))
    results = model.predict(
        source=str(source),
        conf=conf,
        device=device_str(),
        save=save,
        project=str(WORK_DIR / "predict"),
        name="chess",
        exist_ok=True,
        verbose=False,
    )

    all_dets: list[dict[str, Any]] = []
    for result in results:
        dets = parse_detections(result, conf_threshold=conf)
        all_dets.extend(dets)
        print(f"\n[{result.path}] {len(dets)} pieces")
        counts = Counter(d["class_name"] for d in dets)
        for name, n in counts.most_common():
            print(f"  {name}: {n}")
        for d in dets:
            cx, cy = d["center"]
            print(
                f"  - {d['class_name']:16s} "
                f"conf={d['confidence']:.2f} "
                f"center=({cx:.1f}, {cy:.1f})"
            )
    return all_dets


def sample_image_dir() -> Path:
    yaml_path = YOLO_DIR / "data.yaml"
    if yaml_path.exists():
        with open(yaml_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        root = Path(cfg.get("path", YOLO_DIR))
        for rel in ("images/test", "images/val", "images/train"):
            candidate = root / rel
            if candidate.exists() and any(candidate.iterdir()):
                return candidate
    for candidate in (
        YOLO_DIR / "images" / "test",
        YOLO_DIR / "images" / "val",
        YOLO_DIR / "images" / "train",
    ):
        if candidate.exists() and any(candidate.iterdir()):
            return candidate
    return YOLO_DIR / "images" / "val"


def cmd_all(
    extra: bool,
    epochs: int,
    batch: int | None,
    source: str = "auto",
) -> None:
    cmd_prepare(extra=extra, source=source)
    cmd_train(epochs=epochs, batch=batch)
    samples = sorted(sample_image_dir().glob("*"))[:3]
    for img in samples:
        if _is_image(img):
            cmd_predict(img, save=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Chess piece YOLOv8 — robot-arm ready detection",
        epilog=(
            "예: python model_alpha_v2.py prepare\n"
            "    python model_alpha_v2.py train --epochs 30\n"
            "    python model_alpha_v2.py predict --source board.jpg"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", metavar="command")

    prep = sub.add_parser("prepare", help="데이터 다운로드 + YOLO data.yaml 구성")
    prep.add_argument(
        "--extra",
        action="store_true",
        help=f"추가 데이터셋 합치기 ({EXTRA_DATASET})",
    )
    prep.add_argument(
        "--source",
        choices=("auto", "kaggle", "huggingface"),
        default="auto",
        help="auto: Kaggle 우선, 실패 시 HuggingFace",
    )

    tr = sub.add_parser("train", help="YOLOv8 학습")
    tr.add_argument("--epochs", type=int, default=50)
    tr.add_argument("--imgsz", type=int, default=640)
    tr.add_argument("--batch", type=int, default=None)
    tr.add_argument("--patience", type=int, default=10)

    pr = sub.add_parser("predict", help="학습된 모델로 추론 (로봇용 좌표 출력)")
    pr.add_argument("--source", required=True, help="이미지/폴더 경로")
    pr.add_argument("--weights", type=Path, default=None)
    pr.add_argument("--conf", type=float, default=0.25)
    pr.add_argument("--no-save", action="store_true")

    all_p = sub.add_parser("all", help="prepare → train → 샘플 predict")
    all_p.add_argument("--extra", action="store_true")
    all_p.add_argument("--epochs", type=int, default=10)
    all_p.add_argument("--batch", type=int, default=None)
    all_p.add_argument(
        "--source",
        choices=("auto", "kaggle", "huggingface"),
        default="auto",
    )

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    print(f"PyTorch {torch.__version__} | device={device_str()}")

    if args.command == "prepare":
        cmd_prepare(extra=args.extra, source=args.source)
    elif args.command == "train":
        cmd_train(
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            patience=args.patience,
        )
    elif args.command == "predict":
        cmd_predict(
            source=args.source,
            weights=args.weights,
            conf=args.conf,
            save=not args.no_save,
        )
    elif args.command == "all":
        cmd_all(
            extra=args.extra,
            epochs=args.epochs,
            batch=args.batch,
            source=args.source,
        )


if __name__ == "__main__":
    main()
