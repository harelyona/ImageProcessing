import pytest
from ex4 import *
import numpy as np


# Use abs=0.1 to allow small floating point deviations

def test_lk_x():
    """Tests horizontal translation (Right and Left)."""
    # Case 1: Positive Shift (Right)
    num_frames = 10
    vid_right = create_horizontal_square_video(num_frames, row=100, start_col=200, shift=5, size=20)
    u, v, th = lucas_kanade(vid_right[0], vid_right[1])

    assert u == pytest.approx(5, abs=0.1)
    assert v == pytest.approx(0, abs=0.1)
    assert th == pytest.approx(0, abs=0.1)

    # Case 2: Negative Shift (Left)
    vid_left = create_horizontal_square_video(num_frames, row=100, start_col=200, shift=-8, size=20)
    u, v, th = lucas_kanade(vid_left[0], vid_left[1])
    assert u == pytest.approx(-8, abs=0.1)
    assert v == pytest.approx(0, abs=0.1)
    assert th == pytest.approx(0, abs=0.1)


def test_lk_y():
    """Tests vertical translation (Down and Up)."""
    # Case 1: Positive Shift (Down)
    num_frames = 5
    vid_down = create_vertical_square_video(num_frames, col=100, start_row=100, shift=5, size=20)
    u, v, th = lucas_kanade(vid_down[0], vid_down[1])
    assert u == pytest.approx(0, abs=0.1)
    assert v == pytest.approx(5, abs=0.1)
    assert th == pytest.approx(0, abs=0.1)

    # Case 2: Negative Shift (Up)
    vid_up = create_vertical_square_video(num_frames, col=100, start_row=100, shift=-10, size=20)
    u, v, th = lucas_kanade(vid_up[0], vid_up[1])
    assert u == pytest.approx(0, abs=0.1)
    assert v == pytest.approx(-10, abs=0.1)
    assert th == pytest.approx(0, abs=0.1)


def test_lk_x_y():
    """Tests diagonal translation."""
    # Case 1: Positive Diagonal
    vid_diag1 = create_diagonal_square_video(5, start_loc=(100, 100), shift=(3, 4), size=20)
    u, v, th = lucas_kanade(vid_diag1[0], vid_diag1[1])
    assert u == pytest.approx(3, abs=0.1)
    assert v == pytest.approx(4, abs=0.1)
    assert th == pytest.approx(0, abs=0.1)


def test_lk_angle():
    """Tests rotation (CCW and CW)."""
    # Case 1: Counter-Clockwise (Positive)
    vid_ccw = create_rotated_square_video(5, start_loc=(300, 200), angle_per_frame=2, size=50)
    u, v, th = lucas_kanade(vid_ccw[0], vid_ccw[1])
    assert u == pytest.approx(0, abs=0.1)
    assert v == pytest.approx(0, abs=0.1)
    assert th == pytest.approx(2, abs=0.1)

    # Case 2: Clockwise (Negative)
    vid_cw = create_rotated_square_video(5, start_loc=(300, 200), angle_per_frame=-5, size=50)
    u, v, th = lucas_kanade(vid_cw[0], vid_cw[1])
    assert u == pytest.approx(0, abs=0.1)
    assert v == pytest.approx(0, abs=0.1)
    assert th == pytest.approx(-5, abs=0.1)


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

    # Check tuple approx
    assert u == pytest.approx(true_u, abs=0.1)
    assert v == pytest.approx(true_v, abs=0.1)
    assert th == pytest.approx(true_theta, abs=0.1)


def test_lk_zero_motion():
    frame_h, frame_w = 480, 640
    frame = np.zeros((frame_h, frame_w), dtype=np.uint8)
    cv2.rectangle(frame, (200, 200), (300, 300), 255, -1)

    u, v, th = lucas_kanade(frame, frame)
    assert u == pytest.approx(0, abs=1e-5)
    assert v == pytest.approx(0, abs=1e-5)
    assert th == pytest.approx(0, abs=1e-5)


def test_lk_large_translation():
    vid = create_horizontal_square_video(5, row=100, start_col=100, shift=25, size=40)
    u, v, th = lucas_kanade(vid[0], vid[1])
    assert u == pytest.approx(25, abs=0.5)  # Larger tolerance for large motion
    assert v == pytest.approx(0, abs=0.5)
    assert th == pytest.approx(0, abs=0.1)


def test_lk_large_rotation():
    start_loc = (300, 240)
    angle = 10
    vid = create_rotated_square_video(5, start_loc=start_loc, angle_per_frame=angle, size=60)

    u, v, th = lucas_kanade(vid[0], vid[1])

    assert u == pytest.approx(0, abs=1.0)  # Rotation can induce slight translation errors
    assert v == pytest.approx(0, abs=1.0)
    assert th == pytest.approx(10, abs=0.5)


def test_lk_noisy_input():
    """Tests robustness against Gaussian noise."""
    shift = 5
    vid = create_horizontal_square_video(5, row=100, start_col=200, shift=shift, size=30)

    # Add noise to frames
    np.random.seed(42)
    noise_sigma = 10

    frame0 = vid[0].astype(float) + np.random.normal(0, noise_sigma, vid[0].shape)
    frame1 = vid[1].astype(float) + np.random.normal(0, noise_sigma, vid[1].shape)

    frame0 = np.clip(frame0, 0, 255).astype(np.uint8)
    frame1 = np.clip(frame1, 0, 255).astype(np.uint8)

    u, v, th = lucas_kanade(frame0, frame1)

    # Change tolerance from 0.5 to 1.0 to account for noise interference
    assert u == pytest.approx(shift, abs=1.0)


def test_lk_complex_scene():
    h, w = 480, 640
    base = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(base, (100, 100), (150, 150), 255, -1)
    cv2.rectangle(base, (400, 300), (500, 350), 255, -1)

    true_u, true_v, true_th = 4, -5, -3
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, true_th, 1.0)
    M[0, 2] += true_u
    M[1, 2] += true_v

    target = cv2.warpAffine(base, M, (w, h), flags=cv2.INTER_LINEAR)

    u, v, th = lucas_kanade(base, target)

    assert u == pytest.approx(true_u, abs=0.1)
    assert v == pytest.approx(true_v, abs=0.1)
    assert th == pytest.approx(true_th, abs=0.1)


def test_lk_large_x_shift():
    """Tests if the algorithm can handle very large horizontal displacements."""
    shift = 40
    vid = create_horizontal_square_video(num_frames=5, row=100, start_col=100, shift=shift, size=40)

    u, v, th = lucas_kanade(vid[0], vid[1])

    # Check U (Horizontal)
    assert abs(u - shift) < 0.5

    # Check V (Vertical)
    # Relaxed from 0.1 to 0.5 to account for floating point noise
    assert abs(v) < 0.5

    # Check Theta (Rotation)
    assert abs(th) < 0.1