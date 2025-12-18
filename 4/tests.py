import pytest
from ex4 import *


def test_lk_x():
    """Tests horizontal translation (Right and Left)."""
    # Case 1: Positive Shift (Right)
    num_frames = 10
    vid_right = create_horizontal_square_video(num_frames, row=100, start_col=200, shift=5, size=20)
    u, v, th = lucas_kanade(vid_right[0], vid_right[1])
    assert u == 5
    assert v == 0
    assert th == 0

    # Case 2: Negative Shift (Left)
    vid_left = create_horizontal_square_video(num_frames, row=100, start_col=200, shift=-8, size=20)
    u, v, th = lucas_kanade(vid_left[0], vid_left[1])
    assert u == -8
    assert v == 0
    assert th == 0


def test_lk_y():
    """Tests vertical translation (Down and Up)."""
    # Case 1: Positive Shift (Down)
    num_frames = 5
    vid_down = create_vertical_square_video(num_frames, col=100, start_row=100, shift=5, size=20)
    u, v, th = lucas_kanade(vid_down[0], vid_down[1])
    assert u == 0
    assert v == 5
    assert th == 0

    # Case 2: Negative Shift (Up)
    vid_up = create_vertical_square_video(num_frames, col=100, start_row=100, shift=-10, size=20)
    u, v, th = lucas_kanade(vid_up[0], vid_up[1])
    assert u == 0
    assert v == -10
    assert th == 0


def test_lk_x_y():
    """Tests diagonal translation."""
    # Case 1: Positive Diagonal
    vid_diag1 = create_diagonal_square_video(5, start_loc=(100, 100), shift=(3, 4), size=20)
    u, v, th = lucas_kanade(vid_diag1[0], vid_diag1[1])
    assert u == 3
    assert v == 4
    assert th == 0

    # Case 2: Mixed Diagonal (Left + Down)
    vid_diag2 = create_diagonal_square_video(5, start_loc=(200, 100), shift=(-6, 2), size=20)
    u, v, th = lucas_kanade(vid_diag2[0], vid_diag2[1])
    assert u == -6
    assert v == 2
    assert th == 0


def test_lk_angle():
    """Tests rotation (CCW and CW)."""
    # Case 1: Counter-Clockwise (Positive)
    vid_ccw = create_rotated_square_video(5, start_loc=(300, 200), angle_per_frame=2, size=50)
    u, v, th = lucas_kanade(vid_ccw[0], vid_ccw[1])
    assert u == 0
    assert v == 0
    assert th == 2

    # Case 2: Clockwise (Negative)
    vid_cw = create_rotated_square_video(5, start_loc=(300, 200), angle_per_frame=-5, size=50)
    u, v, th = lucas_kanade(vid_cw[0], vid_cw[1])
    assert u == 0
    assert v == 0
    assert th == -5

def test_lk_all_directions():
    """Tests combined Translation and Rotation."""
    frame_h, frame_w = 480, 640
    base_frame = np.zeros((frame_h, frame_w), dtype=np.uint8)
    cv2.rectangle(base_frame, (200, 200), (300, 300), 255, -1)
    center = (frame_w // 2, frame_h // 2)

    # Case 1: Small Shift + Rotation
    true_u, true_v, true_theta = 3, -2, 2
    M1 = cv2.getRotationMatrix2D(center, true_theta, 1.0)
    M1[0, 2] += true_u
    M1[1, 2] += true_v
    target_frame1 = cv2.warpAffine(base_frame, M1, (frame_w, frame_h), flags=cv2.INTER_LINEAR)

    u, v, th = lucas_kanade(base_frame, target_frame1)
    assert (u, v, th) == (true_u, true_v, true_theta)

    # Case 2: Negative Shift + Negative Rotation
    true_u2, true_v2, true_theta2 = -5, 5, -3
    M2 = cv2.getRotationMatrix2D(center, true_theta2, 1.0)
    M2[0, 2] += true_u2
    M2[1, 2] += true_v2
    target_frame2 = cv2.warpAffine(base_frame, M2, (frame_w, frame_h), flags=cv2.INTER_LINEAR)

    u, v, th = lucas_kanade(base_frame, target_frame2)
    assert (u, v, th) == (true_u2, true_v2, true_theta2)


def test_lk_zero_motion():
    """Tests that identical frames return 0,0,0."""
    frame_h, frame_w = 480, 640
    frame = np.zeros((frame_h, frame_w), dtype=np.uint8)
    cv2.rectangle(frame, (200, 200), (300, 300), 255, -1)

    # Pass the same frame twice
    u, v, th = lucas_kanade(frame, frame)
    assert u == 0
    assert v == 0
    assert th == 0


def test_lk_large_translation():
    """Tests if pyramid levels handle large shifts (>15 pixels)."""
    # Shift of 25 pixels is quite large; requires at least 3 pyramid levels
    # Level 0: 25px -> Level 1: 12.5px -> Level 2: 6.25px (solvable)
    vid = create_horizontal_square_video(5, row=100, start_col=100, shift=25, size=40)

    u, v, th = lucas_kanade(vid[0], vid[1])
    assert u == 25
    assert v == 0
    assert th == 0


def test_lk_large_rotation():
    """Tests larger rotation angles (e.g., 10 degrees)."""
    # 10 degrees is significant for iterative solvers
    start_loc = (300, 240)
    angle = 10
    vid = create_rotated_square_video(5, start_loc=start_loc, angle_per_frame=angle, size=60)

    u, v, th = lucas_kanade(vid[0], vid[1])

    assert u == 0
    assert v == 0
    assert th == 10


def test_lk_noisy_input():
    """Tests robustness against Gaussian noise."""
    shift = 5
    vid = create_horizontal_square_video(5, row=100, start_col=200, shift=shift, size=30)

    # Add noise to frames
    np.random.seed(42)  # Fixed seed for reproducibility
    noise_sigma = 10

    # Convert, add noise, clip, convert back
    frame0 = vid[0].astype(float) + np.random.normal(0, noise_sigma, vid[0].shape)
    frame1 = vid[1].astype(float) + np.random.normal(0, noise_sigma, vid[1].shape)

    frame0 = np.clip(frame0, 0, 255).astype(np.uint8)
    frame1 = np.clip(frame1, 0, 255).astype(np.uint8)

    u, v, th = lucas_kanade(frame0, frame1)

    # Should still find the dominant motion despite noise
    assert u == shift
    assert v == 0
    assert th == 0


def test_lk_complex_scene():
    """Tests a scene with multiple objects moving together."""
    h, w = 480, 640
    base = np.zeros((h, w), dtype=np.uint8)

    # Draw two objects (Square and a separate Rectangle)
    cv2.rectangle(base, (100, 100), (150, 150), 255, -1)
    cv2.rectangle(base, (400, 300), (500, 350), 255, -1)

    # Apply global transformation
    true_u, true_v, true_th = 4, -5, -3
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, true_th, 1.0)
    M[0, 2] += true_u
    M[1, 2] += true_v

    target = cv2.warpAffine(base, M, (w, h), flags=cv2.INTER_LINEAR)

    u, v, th = lucas_kanade(base, target)
    assert (u, v, th) == (true_u, true_v, true_th)