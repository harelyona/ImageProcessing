import os
from typing import Tuple
import numpy as np
import cv2
import mediapy as mp

# Constants
FPS = 30
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
BG_COLOR = (0, 0, 0)  # Black
SQUARE_COLOR = (255, 255, 255)  # White
VIDEOS_SAVE_PATH = "videos" + os.sep


def create_square_frame(size: int, upper_left_loc: Tuple[int, int]) -> np.ndarray:
    """
    Creates a single frame with a white square at the specified location.
    """
    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    frame[:] = BG_COLOR

    x, y = upper_left_loc

    # Draw filled rectangle
    cv2.rectangle(frame, (x, y), (x + size, y + size), SQUARE_COLOR, -1)

    return frame


def create_horizontal_square_video(num_frames: int, row: int, start_col: int, shift: int, size: int, save_path: str=None) -> np.ndarray:
    """
    Creates a horizontal video.
    shift: Integer pixels to move per frame (positive for right, negative for left).
    """
    frames = []

    for i in range(num_frames):
        # Exact integer calculation: start + (frame_index * shift)
        current_x = start_col + (i * shift)

        frame = create_square_frame(size=size, upper_left_loc=(current_x, row))
        frames.append(frame)
    if save_path:
        save_video(frames, save_path + ".mp4")
    return np.array(frames)


def create_vertical_square_video(num_frames: int, col: int, start_row: int, shift: int, size: int, save_path: bool=None) -> np.ndarray:
    """
    Creates a vertical video.
    shift: Integer pixels to move per frame (positive for down, negative for up).
    """
    frames = []

    for i in range(num_frames):
        current_y = start_row + (i * shift)

        frame = create_square_frame(size=size, upper_left_loc=(col, current_y))
        frames.append(frame)


    return np.array(frames)


def create_diagonal_square_video(num_frames: int, start_loc: Tuple[int, int], shift: Tuple[int, int],
                                 size: int, save_path: bool=None) -> np.ndarray:
    """
    Creates a diagonal video.
    shift: A tuple (shift_x, shift_y) specifying pixels to move in each axis per frame.
    """
    frames = []
    start_x, start_y = start_loc
    shift_x, shift_y = shift

    for i in range(num_frames):
        current_x = start_x + (i * shift_x)
        current_y = start_y + (i * shift_y)

        frame = create_square_frame(size=size, upper_left_loc=(current_x, current_y))
        frames.append(frame)

    return np.array(frames)


def create_rotated_square_video(num_frames: int, start_loc: Tuple[int, int], angle_per_frame: float,
                                size: int) -> np.ndarray:
    """
    Creates a video where the entire view rotates around the image center.
    This simulates a camera rolling.

    Args:
        angle_per_frame: Degrees to rotate per frame (Positive = Counter-Clockwise).
    """
    frames = []

    # 1. Create the base frame with the square in its initial position
    # We do this once because the object itself isn't moving, the "camera" is.
    base_frame = create_square_frame(size=size, upper_left_loc=start_loc)

    # Center of rotation (Image Center)
    center = (FRAME_WIDTH // 2, FRAME_HEIGHT // 2)

    for i in range(num_frames):
        current_angle = i * angle_per_frame

        # 2. Get the rotation matrix for the current angle
        # center: (x, y), angle: degrees, scale: 1.0
        M = cv2.getRotationMatrix2D(center, current_angle, 1.0)

        # 3. Apply the rotation to the base frame
        # borderValue=BG_COLOR ensures the background stays black when corners rotate in
        rotated_frame = cv2.warpAffine(
            base_frame,
            M,
            (FRAME_WIDTH, FRAME_HEIGHT),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=BG_COLOR
        )

        frames.append(rotated_frame)

    return np.array(frames)


def create_rotated_and_moved_square_video(num_frames: int, start_loc: Tuple[int, int],
                                          shift: Tuple[int, int], angle_per_frame: float,
                                          size: int, save_path: str = None) -> np.ndarray:
    frames = []
    start_x, start_y = start_loc
    shift_x, shift_y = shift

    # Center of rotation (Image Center)
    center = (FRAME_WIDTH // 2, FRAME_HEIGHT // 2)

    for i in range(num_frames):
        current_x = start_x + (i * shift_x)
        current_y = start_y + (i * shift_y)

        temp_frame = create_square_frame(size=size, upper_left_loc=(current_x, current_y))

        current_angle = i * angle_per_frame
        M = cv2.getRotationMatrix2D(center, current_angle, 1.0)

        rotated_frame = cv2.warpAffine(
            temp_frame,
            M,
            (FRAME_WIDTH, FRAME_HEIGHT),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=BG_COLOR
        )

        frames.append(rotated_frame)

    frames = np.array(frames)

    if save_path:
        # FIX: Pass 'frames' (the video data), NOT 'num_frames' (the integer count)
        save_video(frames, save_path + ".mp4")

    return frames


# --- Helper for viewing ---
def play_video(video_ndarray: np.ndarray) -> None:
    if video_ndarray.size == 0:
        return
    delay = int(1000 / FPS)
    print(f"Playing video ({len(video_ndarray)} frames). Press 'q' to exit.")
    for frame in video_ndarray:
        cv2.imshow('Debug Video', frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def save_video(frames: list, filename: str) -> None:
    """
    Saves a list of frames (or numpy array) to an MP4 file.
    """
    if len(frames) == 0:
        return

    # Get dimensions from the first frame
    # Note: numpy shape is (height, width, channels), but OpenCV expects (width, height)
    height, width, _ = frames[0].shape
    size = (width, height)

    # Define the codec (mp4v is standard for .mp4)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, FPS, size)

    for frame in frames:
        out.write(frame)

    out.release()
    print(f"Saved video to: {filename}")

# --- Example Usage ---
if __name__ == "__main__":
    video = create_rotated_square_video(num_frames=150, start_loc=(370, 190), angle_per_frame=5, size=50)
    play_video(video)