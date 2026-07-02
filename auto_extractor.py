import os
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm


RAW_VIDEOS_ROOT = Path("Raw_Videos")
DATA_PATH = Path("MP_Data")
NO_SEQUENCES = 30
SEQUENCE_LENGTH = 30
JITTER_STD = 0.003


def extract_keypoints(results) -> np.ndarray:
    """Return exactly 126 values: left hand (63) + right hand (63)."""
    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3)
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            coords = np.array(
                [[res.x, res.y, res.z] for res in hand_landmarks.landmark]
            ).flatten()
            if handedness.classification[0].label == "Right":
                rh = coords
            else:
                lh = coords
    return np.concatenate([lh, rh])


def sample_frame_indices(total_frames: int, target_frames: int = SEQUENCE_LENGTH) -> np.ndarray:
    """Evenly sample exactly target_frames indices from the full video length."""
    if total_frames <= 0:
        return np.zeros(target_frames, dtype=np.int32)
    return np.linspace(0, max(total_frames - 1, 0), target_frames, dtype=np.int32)


def load_video_keypoints(video_path: Path, hands) -> list[np.ndarray]:
    """
    Read full video once and return a list of extracted keypoint vectors (126 each).
    """
    cap = cv2.VideoCapture(str(video_path))
    keypoint_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        keypoint_frames.append(extract_keypoints(results))

    cap.release()
    return keypoint_frames


def process_action_video(action_name: str, video_path: Path, hands) -> None:
    """
    Generate 30 sequences from a single source video by adding slight jitter.
    Saves to MP_Data/action_name/sequence/frame.npy
    """
    keypoint_frames = load_video_keypoints(video_path, hands)
    total_frames = len(keypoint_frames)
    if total_frames == 0:
        raise RuntimeError(f"Video has no readable frames: {video_path}")

    sampled_indices = sample_frame_indices(total_frames, SEQUENCE_LENGTH)

    action_dir = DATA_PATH / action_name
    action_dir.mkdir(parents=True, exist_ok=True)

    for sequence in tqdm(
        range(NO_SEQUENCES),
        desc=f"Augmenting {action_name}",
        leave=False,
    ):
        rng = np.random.default_rng(sequence)
        seq_dir = action_dir / str(sequence)
        seq_dir.mkdir(parents=True, exist_ok=True)

        for frame_num, frame_idx in enumerate(sampled_indices):
            base_keypoints = keypoint_frames[int(frame_idx)].astype(np.float32, copy=True)

            # Add light jitter so each sequence is slightly different.
            noise = rng.normal(loc=0.0, scale=JITTER_STD, size=base_keypoints.shape).astype(
                np.float32
            )
            augmented_keypoints = base_keypoints + noise

            # Keep exact keypoint length guarantee.
            if augmented_keypoints.shape[0] != 126:
                augmented_keypoints = np.zeros(126, dtype=np.float32)

            np.save(seq_dir / f"{frame_num}.npy", augmented_keypoints)


def main() -> None:
    if not RAW_VIDEOS_ROOT.exists():
        print(f"Raw videos folder not found: {RAW_VIDEOS_ROOT}")
        return

    action_dirs = sorted([p for p in RAW_VIDEOS_ROOT.iterdir() if p.is_dir()])
    if not action_dirs:
        print(f"No action folders found in: {RAW_VIDEOS_ROOT}")
        return

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        for action_dir in tqdm(action_dirs, desc="Processing actions"):
            action_name = action_dir.name
            videos = sorted(action_dir.glob("*.mp4"))
            if not videos:
                print(f"Skipping {action_name}: no .mp4 found.")
                continue

            # Use the first .mp4 per action folder (expected workflow: 1 video per word).
            source_video = videos[0]
            try:
                process_action_video(action_name, source_video, hands)
            except Exception as exc:
                print(f"Failed {action_name} ({source_video.name}): {exc}")

    print("Auto extraction complete.")


if __name__ == "__main__":
    main()
