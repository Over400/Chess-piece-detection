"""
듀얼 웹캠 체스 로봇 비전 파이프라인 (설계 골격)

카메라 구성 (타 PC 예정):
  - fixed   : 거치대 고정 웹캠 → 보드 전체(위/대각) → 국면 모니터링·변화 감지
  - gripper : 그리퍼 근처 웹캠 → 집기/놓기 직전·직후 국소 확인

동작 흐름:
  MONITOR  → fixed cam으로 전체 보드 상태 주기적 추정
  CHANGED  → 이전 스냅샷과 diff → 변화 칸·기물 추정
  DECIDE   → (외부) 체스 규칙/엔진 → from/to square
  EXECUTE  → 로봇 이동 + gripper cam으로 목표 칸·기물 검증
  VERIFY   → fixed cam으로 보드 상태 재확인 → MONITOR

model_alpha_v2.py:
  - prepare/train → best.pt (기물 YOLO)
  - predict / parse_detections → Detection dict 형식

cobot2_ws (향후):
  - fixed/gripper 이미지를 ROS Image 토픽으로 수신
  - robot_control은 칸 좌표 또는 SrvDepthPosition 확장 서비스 호출

사용 예 (설계 확인):
  python chess_dual_cam_pipeline.py --demo
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# model_alpha_v2와 동일한 가중치·클래스 (학습 산출물)
ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "runs" / "detect" / "chess_pieces" / "weights" / "best.pt"


class CameraRole(enum.Enum):
    FIXED = "fixed"       # 거치대 — 전체 보드
    GRIPPER = "gripper"   # 그리퍼 근처 — 국소 검증


class PipelinePhase(enum.Enum):
    MONITOR = "monitor"
    CHANGED = "changed"
    DECIDE = "decide"
    EXECUTE = "execute"
    VERIFY = "verify"


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    xyxy: list[float]
    center: tuple[float, float]
    square: str | None = None  # 보드 warp 후 e.g. "e4"


@dataclass
class BoardSnapshot:
    camera: CameraRole
    timestamp: float
    detections: list[Detection]
    # 8x8 각 칸의 기물 class_name 또는 None (fixed cam 전용)
    grid: dict[str, str | None] = field(default_factory=dict)

    def piece_count(self) -> int:
        return len(self.detections)


@dataclass
class BoardChange:
    """fixed cam 두 스냅샷 간 변화."""
    added: dict[str, str]      # square -> class_name
    removed: dict[str, str]
    moved: list[tuple[str, str, str]]  # (from_sq, to_sq, class_name)


def pixel_to_square(
    cx: float,
    cy: float,
    homography: np.ndarray,
) -> str:
    """
    픽셀 중심 → 보드 칸 (a1~h8).
    homography: 4점 코너 캘리브레이션 후 cv2.getPerspectiveTransform 결과.
    TODO: 타 PC에서 보드 코너 라벨링/캘리브 후 주입.
    """
    pt = np.array([[[cx, cy]]], dtype=np.float32)
    board_xy = cv2_perspective_transform(homography, pt)[0][0]
    file_idx = int(np.clip(board_xy[0], 0, 7.999))
    rank_idx = int(np.clip(board_xy[1], 0, 7.999))
    files = "abcdefgh"
    return f"{files[file_idx]}{rank_idx + 1}"


def cv2_perspective_transform(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.perspectiveTransform(pts, h)


def detections_to_grid(
    detections: list[Detection],
    homography: np.ndarray | None,
) -> dict[str, str | None]:
    """detection 목록 → {square: class_name}. homography 없으면 square 미매핑."""
    grid: dict[str, str | None] = {f"{f}{r}": None for f in "abcdefgh" for r in range(1, 9)}
    if homography is None:
        return grid
    for det in detections:
        sq = pixel_to_square(det.center[0], det.center[1], homography)
        det.square = sq
        # 한 칸에 여러 box면 confidence 높은 쪽 (간단 정책)
        if grid.get(sq) is None or det.confidence > 0.5:
            grid[sq] = det.class_name
    return grid


def diff_board_states(
    before: BoardSnapshot,
    after: BoardSnapshot,
) -> BoardChange | None:
    """fixed cam grid diff. 변화 없으면 None."""
    if not before.grid or not after.grid:
        return None

    added: dict[str, str] = {}
    removed: dict[str, str] = {}
    moved: list[tuple[str, str, str]] = []

    for sq in before.grid:
        b, a = before.grid.get(sq), after.grid.get(sq)
        if b == a:
            continue
        if b and not a:
            removed[sq] = b
        elif a and not b:
            added[sq] = a

    # 단순 이동 추정: removed 1 + added 1 + 같은 class → moved
    if len(removed) == 1 and len(added) == 1:
        (fs, fc), (ts, tc) = next(iter(removed.items())), next(iter(added.items()))
        if fc == tc:
            return BoardChange(added={}, removed={}, moved=[(fs, ts, fc)])

    if not added and not removed:
        return None
    return BoardChange(added=added, removed=removed, moved=moved)


class YoloDetector:
    """model_alpha_v2 best.pt 래퍼."""

    def __init__(self, weights: Path = DEFAULT_WEIGHTS, conf: float = 0.25):
        if not weights.exists():
            raise FileNotFoundError(
                f"가중치 없음: {weights}\n"
                "먼저: python model_alpha_v2.py train"
            )
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.conf = conf

    def run(self, image: np.ndarray) -> list[Detection]:
        results = self.model.predict(source=image, conf=self.conf, verbose=False)
        out: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            names = result.names
            for box in result.boxes:
                conf = float(box.conf[0].cpu().item())
                cls_id = int(box.cls[0].cpu().item())
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                out.append(
                    Detection(
                        class_id=cls_id,
                        class_name=names[cls_id],
                        confidence=conf,
                        xyxy=[x1, y1, x2, y2],
                        center=(cx, cy),
                    )
                )
        return out


class DualCamChessPipeline:
    """
    fixed cam: 전체 보드 모니터링 + 변화 트리거
    gripper cam: EXECUTE 단계에서 목표 기물/칸 확인
    """

    def __init__(
        self,
        detector: YoloDetector | None = None,
        fixed_homography: np.ndarray | None = None,
        monitor_interval_s: float = 0.5,
    ):
        self.detector = detector or YoloDetector()
        self.fixed_h = fixed_homography
        self.monitor_interval_s = monitor_interval_s
        self.phase = PipelinePhase.MONITOR
        self._last_snapshot: BoardSnapshot | None = None
        self._pending_change: BoardChange | None = None

    def process_fixed_frame(self, frame: np.ndarray) -> BoardSnapshot:
        dets = self.detector.run(frame)
        grid = detections_to_grid(dets, self.fixed_h)
        snap = BoardSnapshot(
            camera=CameraRole.FIXED,
            timestamp=time.time(),
            detections=dets,
            grid=grid,
        )

        if self.phase == PipelinePhase.MONITOR and self._last_snapshot is not None:
            change = diff_board_states(self._last_snapshot, snap)
            if change is not None:
                self._pending_change = change
                self.phase = PipelinePhase.CHANGED
                print(f"[pipeline] board change detected: {change}")

        self._last_snapshot = snap
        return snap

    def process_gripper_frame(
        self,
        frame: np.ndarray,
        expected_class: str | None = None,
    ) -> tuple[list[Detection], bool]:
        """
        EXECUTE/VERIFY: gripper cam에서 기물 확인.
        expected_class가 있으면 해당 class detection 존재 여부 반환.
        """
        dets = self.detector.run(frame)
        if expected_class is None:
            return dets, True
        ok = any(d.class_name == expected_class for d in dets)
        return dets, ok

    def on_move_decided(self, from_sq: str, to_sq: str, piece: str) -> None:
        """외부 체스 엔진/규칙에서 호출 → EXECUTE 진입."""
        self.phase = PipelinePhase.EXECUTE
        self._move_plan = (from_sq, to_sq, piece)
        print(f"[pipeline] execute: {piece} {from_sq} → {to_sq}")

    def on_execute_done(self, success: bool) -> None:
        self.phase = PipelinePhase.VERIFY if success else PipelinePhase.MONITOR
        if success:
            print("[pipeline] verify with fixed cam after move")

    def reset_monitor(self) -> None:
        self.phase = PipelinePhase.MONITOR
        self._pending_change = None

    @property
    def pending_change(self) -> BoardChange | None:
        return self._pending_change


def _demo() -> None:
    """homography 없이 파이프라인 상태 전환만 출력."""
    print("=== Dual-cam chess pipeline (demo, no camera) ===\n")
    print("Cameras:")
    print("  [fixed]   stand mount — full board, change detection")
    print("  [gripper] on arm      — pick/place verification\n")
    print("Phases:", " → ".join(p.value for p in PipelinePhase))
    print("\nNext steps on target PC:")
    print("  1. Calibrate fixed cam → homography (4 board corners)")
    print("  2. Fine-tune best.pt with your webcam images (model_alpha_v2.py train)")
    print("  3. ROS2: Image topics → process_fixed_frame / process_gripper_frame")
    print("  4. robot_control: square → board mm → movej/movel")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Dual webcam chess pipeline skeleton")
    p.add_argument("--demo", action="store_true", help="Print architecture (no GPU)")
    p.add_argument("--image", type=Path, help="Test fixed-cam frame with YOLO")
    args = p.parse_args()

    if args.demo:
        _demo()
    elif args.image:
        import cv2

        img = cv2.imread(str(args.image))
        if img is None:
            raise SystemExit(f"Cannot read {args.image}")
        pipe = DualCamChessPipeline()
        snap = pipe.process_fixed_frame(img)
        print(f"detections: {snap.piece_count()}")
        for d in snap.detections[:10]:
            print(f"  {d.class_name} conf={d.confidence:.2f} center={d.center}")
    else:
        p.print_help()
