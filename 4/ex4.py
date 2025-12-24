import os
from typing import Tuple, Any, List
import numpy as np
import cv2
from matplotlib import pyplot as plt
from numpy import floating, complexfloating, timedelta64
from openpyxl.styles.alignment import horizontal_alignments
from mediapy import read_video, show_video
from scipy.signal import convolve2d
from square_video import *

PYRAMID_LEVELS = 6
FILTER_SIZE = 5
ITERATIONS_PER_LEVEL = 30
ITERATION_ROTATION = 10

def show_image(img, save_path=None):
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    if save_path:
        plt.imsave(save_path, img, cmap='gray')
    plt.show()

def unite_pyramid_channels(ch1, ch2, ch3):
    united_pyr = []
    for l1, l2, l3 in zip(ch1, ch2, ch3):
        united_pyr.append(np.dstack((l1, l2, l3)))
    return united_pyr


def build_gaussian_pyramid(im, max_levels, filter_size):
    # Normalize if needed
    if im.max() > 1.0 or im.dtype == np.uint8:
        im = im.astype(float) / 255.0

    if im.ndim == 2:
        return build_single_channel_gaussian_pyramid(im, max_levels, filter_size)
    ch1_pyr = build_single_channel_gaussian_pyramid(im[:, :, 0], max_levels, filter_size)
    ch2_pyr = build_single_channel_gaussian_pyramid(im[:, :, 1], max_levels, filter_size)
    ch3_pyr = build_single_channel_gaussian_pyramid(im[:, :, 2], max_levels, filter_size)
    return unite_pyramid_channels(ch1_pyr, ch2_pyr, ch3_pyr)

def generate_gaussian_kernel(kernel_size):
    if kernel_size == 1: return np.array([[1]])
    kernel_1d = (np.poly1d([1, 1]) ** (kernel_size - 1)).c
    kernel_1d = kernel_1d / kernel_1d.sum()
    return kernel_1d.reshape(1, -1)


def expand(image, filter_size):
    """ Upsamples and blurs. Handles 2D and 3D images. """
    kernel_row = generate_gaussian_kernel(filter_size)
    kernel_col = kernel_row.T

    if image.ndim == 3:
        h, w, c = image.shape
        out_shape = (h * 2, w * 2, c)
        expanded_im = np.zeros(out_shape)
        expanded_im[::2, ::2, :] = image
        output_channels = []
        for ch in range(c):
            channel_data = expanded_im[:, :, ch]
            blur_col = convolve2d(channel_data, kernel_col * 2, mode="same", boundary="symm")
            blur_im = convolve2d(blur_col, kernel_row * 2, mode="same", boundary="symm")
            output_channels.append(blur_im)
        return np.dstack(output_channels)
    else:
        out_shape = (image.shape[0] * 2, image.shape[1] * 2)
        expanded_im = np.zeros(out_shape)
        expanded_im[::2, ::2] = image
        blur_col = convolve2d(expanded_im, kernel_col * 2, mode="same", boundary="symm")
        blur_im = convolve2d(blur_col, kernel_row * 2, mode="same", boundary="symm")
        return blur_im

def build_single_channel_gaussian_pyramid(im, max_levels, filter_size):
    kernel_1d = generate_gaussian_kernel(filter_size)
    pyr = [im]
    current_im = im
    for _ in range(max_levels - 1):
        blurred_im = convolve2d(convolve2d(current_im, kernel_1d.T, mode="same", boundary="symm"), kernel_1d,
                                mode="same", boundary="symm")
        downsampled_im = blurred_im[::2, ::2]
        if downsampled_im.shape[0] < 2 or downsampled_im.shape[1] < 2: break
        pyr.append(downsampled_im)
        current_im = downsampled_im
    return pyr

def build_single_channel_laplacian_pyramid(image, max_levels, filter_size):
    gaussian_pyr = build_single_channel_gaussian_pyramid(image, max_levels, filter_size)
    laplacian_pyr = []
    for i in range(len(gaussian_pyr) - 1):
        g_current = gaussian_pyr[i]
        g_next = gaussian_pyr[i + 1]
        expanded_next = expand(g_next, filter_size)

        # Crop if sizes don't match
        if expanded_next.shape != g_current.shape:
            expanded_next = expanded_next[:g_current.shape[0], :g_current.shape[1]]

        laplacian_pyr.append(g_current - expanded_next)

    laplacian_pyr.append(gaussian_pyr[-1])
    return laplacian_pyr

def reconstruct_from_laplacian_pyramid(pyr, filter_size):
    current_im = pyr[-1]
    for i in range(len(pyr) - 2, -1, -1):
        expanded_im = expand(current_im, filter_size)
        laplacian_level = pyr[i]

        if expanded_im.shape[0] != laplacian_level.shape[0] or \
                expanded_im.shape[1] != laplacian_level.shape[1]:
            expanded_im = expanded_im[:laplacian_level.shape[0], :laplacian_level.shape[1]]

        current_im = expanded_im + laplacian_level
    return current_im


def build_laplacian_pyramid(image, max_levels, filter_size):
    if image.max() > 1.0 or image.dtype == np.uint8:
        image = image.astype(float) / 255.0

    if image.ndim == 2:
        return build_single_channel_laplacian_pyramid(image, max_levels, filter_size)

    ch1_pyr = build_single_channel_laplacian_pyramid(image[:, :, 0], max_levels, filter_size)
    ch2_pyr = build_single_channel_laplacian_pyramid(image[:, :, 1], max_levels, filter_size)
    ch3_pyr = build_single_channel_laplacian_pyramid(image[:, :, 2], max_levels, filter_size)

    return unite_pyramid_channels(ch1_pyr, ch2_pyr, ch3_pyr)


def image_blending(im1, im2, mask, max_levels, filter_size):
    # 1. Build Laplacian pyramids (Raw floats)
    L1 = build_laplacian_pyramid(im1, max_levels, filter_size)
    L2 = build_laplacian_pyramid(im2, max_levels, filter_size)

    # 2. Build Gaussian pyramid for mask
    Gm = build_gaussian_pyramid(mask, max_levels, filter_size)

    blended_pyr = []
    for l1, l2, gm in zip(L1, L2, Gm):
        # Handle broadcasting if mask is 2D and images are 3D
        if gm.ndim == 2 and l1.ndim == 3:
            gm = gm[:, :, np.newaxis]

        blended_level = gm * l1 + (1 - gm) * l2
        blended_pyr.append(blended_level)

    im_blended = reconstruct_from_laplacian_pyramid(blended_pyr, filter_size)
    return np.clip(im_blended, 0, 1)

def lk_prep_image(im):
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.dtype != np.float32:
        im = im.astype(np.float32)
    # Normalize to 0-1 if in 0-255 range for stability
    if im.max() > 1.0:
        im /= 255.0
    return im


def lk_for_x_y(frame1: np.ndarray, frame2: np.ndarray) -> Tuple[int, int]:
    """
    Computes the global translation (u, v) from frame1 to frame2.
    Returns INTEGER shifts (rounded from sub-pixel accuracy).
    """
    I1 = lk_prep_image(frame1)
    I2 = lk_prep_image(frame2)

    pyr1 = build_gaussian_pyramid(I1, PYRAMID_LEVELS, FILTER_SIZE)
    pyr2 = build_gaussian_pyramid(I2, PYRAMID_LEVELS, FILTER_SIZE)

    u, v = 0.0, 0.0

    # Coarse-to-fine iterative refinement
    for level in range(len(pyr1) - 1, -1, -1):
        u *= 2
        v *= 2

        im1_lvl = pyr1[level]
        im2_lvl = pyr2[level]
        h, w = im1_lvl.shape

        for _ in range(ITERATIONS_PER_LEVEL):
            # Warp Image 1 using current estimate
            M = np.float32([[1, 0, -u], [0, 1, -v]])
            im1_warp = cv2.warpAffine(im1_lvl, M, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

            # Compute Gradients on the WARPED image
            Ix = cv2.Sobel(im1_warp, cv2.CV_64F, 1, 0, ksize=3)
            Iy = cv2.Sobel(im1_warp, cv2.CV_64F, 0, 1, ksize=3)
            It = im1_warp - im2_lvl

            # Linear System A * d = b
            Ixx = np.sum(Ix * Ix)
            Iyy = np.sum(Iy * Iy)
            Ixy = np.sum(Ix * Iy)
            Ixt = np.sum(Ix * It)
            Iyt = np.sum(Iy * It)

            A = np.array([[Ixx, Ixy], [Ixy, Iyy]])
            b = np.array([[Ixt], [Iyt]])

            try:
                delta = np.linalg.pinv(A) @ b
                du, dv = delta.flatten()
                u += du
                v += dv

                if np.abs(du) < 1e-5 and np.abs(dv) < 1e-5:
                    break
            except np.linalg.LinAlgError:
                break

    # Return rounded integers
    return int(round(u)), int(round(v))


def find_rotation_angle(I1: np.ndarray, I2: np.ndarray) -> int:
    """
    Estimates the rotation angle (theta) from I1 to I2.
    Uses a Pyramid and Joint Solver (u, v, theta) to robustly distinguish
    true rotation from large translations.
    """

    # Build Pyramids
    pyr1 = build_gaussian_pyramid(I1, PYRAMID_LEVELS, FILTER_SIZE)
    pyr2 = build_gaussian_pyramid(I2, PYRAMID_LEVELS, FILTER_SIZE)

    # State: u, v, theta
    u, v, theta = 0.0, 0.0, 0.0

    # Coarse-to-fine loop
    for level in range(len(pyr1) - 1, -1, -1):
        # Scale up translation (theta is scale-invariant)
        u *= 2
        v *= 2

        im1_lvl = pyr1[level]
        im2_lvl = pyr2[level]
        h, w = im1_lvl.shape
        cy, cx = h / 2.0, w / 2.0

        # Grid for rotation derivatives (Centered)
        y_grid, x_grid = np.mgrid[0:h, 0:w]
        x_grid = x_grid.astype(np.float32) - cx
        y_grid = y_grid.astype(np.float32) - cy

        for _ in range(ITERATION_ROTATION):
            # 1. Construct Matrix (Rotate then Translate)
            M = cv2.getRotationMatrix2D((cx, cy), theta, 1.0)
            M[0, 2] -= u
            M[1, 2] -= v

            # 2. Warp I1 towards I2 using Inverse Map
            im1_warp = cv2.warpAffine(
                im1_lvl, M, (w, h),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

            # 3. Error & Gradients
            It = im1_warp - im2_lvl

            # Gradients on Warped Image
            Ix = cv2.Sobel(im1_warp, cv2.CV_64F, 1, 0, ksize=3) / 8.0
            Iy = cv2.Sobel(im1_warp, cv2.CV_64F, 0, 1, ksize=3) / 8.0

            # 4. Jacobians
            # J_u = -Ix, J_v = -Iy
            # J_theta (Calculated) = x*Iy - y*Ix
            # Theoretical Deriv dI/dTh = y*Ix - x*Iy
            # So J_theta here is effectively -dI/dTh (Negative Derivative)
            J_theta = (x_grid * Iy - y_grid * Ix) * (np.pi / 180.0)

            # 5. Build System
            Ix_f = Ix.flatten()
            Iy_f = Iy.flatten()
            Jth_f = J_theta.flatten()
            It_f = It.flatten()

            Ixx = np.dot(Ix_f, Ix_f)
            Iyy = np.dot(Iy_f, Iy_f)
            Itt = np.dot(Jth_f, Jth_f)
            Ixy = np.dot(Ix_f, Iy_f)
            Ixt = np.dot(Ix_f, Jth_f)
            Iyt = np.dot(Iy_f, Jth_f)

            A = np.array([
                [Ixx, Ixy, Ixt],
                [Ixy, Iyy, Iyt],
                [Ixt, Iyt, Itt]
            ])

            # b Calculation
            # We need b = - (Jacobian * Error)
            # b_u = - sum(-Ix * It) = sum(Ix * It)
            # b_v = - sum(-Iy * It) = sum(Iy * It)
            # b_th = - sum(J_theta * It)  <-- Wait!
            # Since J_theta is ALREADY negative derivative,
            # - (J_theta * It) would be Positive Gradient. We want Negative Gradient.
            # So we strictly want: b_th = sum(J_theta * It)

            b = np.array([
                np.dot(Ix_f, It_f),
                np.dot(Iy_f, It_f),
                np.dot(Jth_f, It_f)  # [CORRECTION] Removed the negative sign here
            ])

            try:
                # Use pinv for stability on featureless images (like the square)
                delta = np.linalg.pinv(A) @ b
                du, dv, dtheta = delta

                # Additive Update
                u += du
                v += dv
                theta += dtheta

                if abs(du) < 1e-3 and abs(dv) < 1e-3 and abs(dtheta) < 1e-3:
                    break
            except np.linalg.LinAlgError:
                break

    # Return only the rotation component
    return -int(round(theta))


def lucas_kanade(frame1: np.ndarray, frame2: np.ndarray) -> Tuple[int, int, int]:
    """
    1. Finds rotation theta using a pyramid (robust to large shifts).
    2. Rotates frame2 by -theta to align with frame1.
    3. Uses lk_for_x_y to find translation (u, v).
    """
    I1 = lk_prep_image(frame1)
    I2 = lk_prep_image(frame2)

    # Step 1: Find Rotation Angle
    theta = find_rotation_angle(I1, I2)

    # Step 2: Compensate for Rotation
    h, w = I1.shape
    cx, cy = w / 2.0, h / 2.0

    # Rotate frame2 by -theta to cancel out the detected rotation
    M_fix = cv2.getRotationMatrix2D((cx, cy), -theta, 1.0)

    # Use standard forward warp to apply the fix
    I2_aligned = cv2.warpAffine(
        I2, M_fix, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # Step 3: Find Translation
    u, v = lk_for_x_y(I1, I2_aligned)

    return int(u), int(v), int(round(theta))


def align_frames(video: np.ndarray, shifts_path: str = None):
    """
    Step 2: Stabilize Rotations & Y translations.
    Returns a video where the only motion left is horizontal (X).
    """
    h, w = video.shape[1], video.shape[2]
    num_frames = video.shape[0]

    # 1. Load or Calculate Shifts
    if shifts_path:
        data = np.load(shifts_path)
        dx, dy, dtheta = data['dx'], data['dy'], data['dtheta']
    else:
        dx, dy, dtheta = get_video_shifts(video)  # Assumes this function is defined
        # Optional: Save them for next time
        # np.savez("calculated_shifts.npz", dx=dx, dy=dy, dtheta=dtheta)

    # 2. Calculate "Tall" Canvas Height
    # We need enough vertical space so the frames don't go out of bounds when we shift them up/down.
    total_dy = int(np.sum(np.abs(dy)))
    aligned_h = h + total_dy + 200  # +200 buffer

    # 3. Initialize Loop Variables
    # Start placing frames in the middle of the new tall canvas
    current_y = total_dy // 2 + 100
    current_angle = 0.0

    # We rotate around the original image center
    center_x, center_y = w // 2, h // 2

    aligned_frames = []
    for i in range(num_frames):
        frame = video[i]

        # --- A. Build Transformation Matrix ---

        # 1. Rotation Matrix (around center)
        # Note: We negate current_angle to stabilize (counter-act the motion)
        M = cv2.getRotationMatrix2D((center_x, center_y), -current_angle, 1.0)

        # 2. Add Vertical Translation (Stabilize Y)
        # The rotation matrix is 2x3: [[cos, -sin, tx], [sin, cos, ty]]
        # We need to shift the image so it lands at 'current_y' in the new tall canvas.
        # Currently, the center of the image is at h//2.
        # We want it to be at 'current_y'.
        # So shift = current_y - (h // 2)
        y_shift = current_y - (h // 2)

        # Add this shift to the translation component (row 1, col 2)
        M[1, 2] += y_shift

        # --- B. Warp Frame ---
        # Apply the transform. The output size is (w, aligned_h).
        warped_frame = cv2.warpAffine(
            frame,
            M,
            (w, aligned_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        aligned_frames.append(warped_frame)

        # --- C. Update Accumulators for NEXT frame ---
        if i < num_frames - 1:
            current_angle += dtheta[i + 1]
            current_y -= int(round(dy[i + 1]))

    return np.array(aligned_frames), dx


def create_empty_panorama(video: np.ndarray) -> np.ndarray:
    """
    Computes the size of the panorama needed to fit all frames based on the shifts.
    Returns (min_x, max_x, min_y, max_y).
    """
    frame_hight, frame_width = video.shape[1], video.shape[2]
    xshifts, yshifts, _ = get_video_shifts(video)
    cum_xshifts = np.cumsum(xshifts)
    cum_yshifts = np.cumsum(yshifts)
    panorama_size = frame_hight + cum_xshifts[-1], frame_width + cum_yshifts[-1]
    return np.zeros(panorama_size, dtype=video.dtype)


def get_video_shifts(video: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes (dx, dy, dtheta) between consecutive frames using Lucas-Kanade.
    """
    num_frames = video.shape[0]
    x_shifts = np.zeros(num_frames)
    y_shifts = np.zeros(num_frames)
    th_shifts = np.zeros(num_frames)

    # Calculate shifts for each pair of frames
    for i in range(num_frames - 1):
        x_shifts[i + 1], y_shifts[i + 1], th_shifts[i + 1] = lucas_kanade(video[i], video[i + 1])

    return x_shifts, y_shifts, th_shifts


def get_panorama_matrices(dx, dy, dtheta, h, w):
    transforms = []
    cx, cy = w / 2.0, h / 2.0

    for i in range(len(dx)):
        # 1. Reverse Rotation (Frame i+1 -> i)
        M_rot = cv2.getRotationMatrix2D((cx, cy), -dtheta[i], 1.0)
        M_rot = np.vstack([M_rot, [0, 0, 1]])

        # 2. Reverse Translation (Frame i+1 -> i)
        M_trans = np.eye(3)
        M_trans[0, 2] = -dx[i]
        M_trans[1, 2] = -dy[i]

        transforms.append(M_trans @ M_rot)

    return transforms


def align_to_middle_frame(frames, motion_matrices):
    """
    Aligns all frames to the coordinate system of the middle frame.

    Args:
        frames: List of images.
        motion_matrices: List where motion_matrices[i] transforms frame[i+1] to frame[i].
                         Length must be len(frames) - 1.
    """
    num_frames = len(frames)
    mid_idx = num_frames // 2

    # Initialize list of global transforms (one per frame)
    # We fill it with None first to assign by index
    global_transforms = [None] * num_frames

    # The middle frame is our anchor (Identity)
    global_transforms[mid_idx] = np.eye(3)

    # 1. Chain BACKWARDS from Middle to Start (0)
    # motion_matrices[i] is transform for frame[i+1] -> frame[i]
    # To go from i -> i+1 (which is moving towards the middle anchor), we need Inverse.
    current_transform = np.eye(3)

    for i in range(mid_idx - 1, -1, -1):
        # We want transform: Frame i -> Middle
        # We have motion M: Frame i+1 -> Frame i
        # We know T: Frame i+1 -> Middle
        # Therefore: T_new = T * M_inverse

        M = motion_matrices[i]
        M_inv = np.linalg.inv(M)

        current_transform = current_transform @ M_inv
        global_transforms[i] = current_transform

    # 2. Chain FORWARDS from Middle to End
    current_transform = np.eye(3)

    for i in range(mid_idx, num_frames - 1):
        # We want transform: Frame i+1 -> Middle
        # We have motion M: Frame i+1 -> Frame i
        # We know T: Frame i -> Middle
        # Therefore: T_new = T * M

        M = motion_matrices[i]

        current_transform = current_transform @ M
        global_transforms[i + 1] = current_transform

    # --- From here, the warping logic is identical to the previous version ---

    # 3. Calculate Canvas Size (Bounding Box)
    h, w = frames[0].shape[:2]
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
    all_corners = []

    for H in global_transforms:
        warped_corners = cv2.perspectiveTransform(corners, H)
        all_corners.append(warped_corners)

    all_corners = np.concatenate(all_corners, axis=0)

    [x_min, y_min] = all_corners.min(axis=0).ravel()
    [x_max, y_max] = all_corners.max(axis=0).ravel()

    translation_dist = [-x_min, -y_min]

    H_translation = np.array([
        [1, 0, translation_dist[0]],
        [0, 1, translation_dist[1]],
        [0, 0, 1]
    ])

    # 4. Warp Images
    warped_frames = []
    output_width = int(x_max - x_min)
    output_height = int(y_max - y_min)

    for i, frame in enumerate(frames):
        H_final = H_translation @ global_transforms[i]

        warped = cv2.warpPerspective(
            frame,
            H_final,
            (output_width, output_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        warped_frames.append(warped)

    return np.array(warped_frames), (output_height, output_width)


def stabilize_video(frames: np.ndarray, dy: np.ndarray, dtheta: np.ndarray) -> np.ndarray:
    """
    Phase 1: Aligns all frames to a common coordinate system (Stabilization).
    Implements step 2: "Stabilize Rotations & Y translations".
    Returns a 'tall' video where the horizon is stable.
    """
    h, w = frames[0].shape[:2]
    num_frames = len(frames)

    # 1. Calculate 'Tall' Canvas Height
    # We need enough vertical space so frames don't leave the canvas when stabilized.
    total_dy = int(np.sum(np.abs(dy)))
    aligned_h = h + total_dy + 200  # generous buffer

    center_x, center_y = w // 2, h // 2

    # Start placing frames in the middle of the tall canvas
    current_y = aligned_h // 2
    current_angle = 0.0

    stabilized_frames = []

    print("Phase 1: Stabilizing video...")
    for i in range(num_frames):
        if i == 268:
            pass
        # --- Build Transformation Matrix ---
        # 1. Rotation (Inverse of accumulated angle)
        M = cv2.getRotationMatrix2D((center_x, center_y), -current_angle, 1.0)

        # 2. Vertical Translation (Stabilize Y)
        # Shift so the image center (h//2) lands at the stabilized Y position (current_y)
        y_shift = current_y - (h // 2)
        M[1, 2] += y_shift

        # --- Warp ---
        warped_frame = cv2.warpAffine(
            frames[i],
            M,
            (w, aligned_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        stabilized_frames.append(warped_frame)

        # --- Update Accumulators for NEXT frame ---
        if i < num_frames - 1:
            current_angle += dtheta[i + 1]
            current_y -= int(round(dy[i + 1]))

    return np.array(stabilized_frames)


def strip_stitching(frames: np.ndarray, dx: np.ndarray, dy: np.ndarray, dtheta: np.ndarray, k: int) -> np.ndarray:
    """
    Phase 2: Stitching without manual stretch factor.
    Relies on accurate dx values from improved Lucas-Kanade.
    """
    h, w = frames[0].shape[:2]

    # --- PHASE 1: ALIGNMENT ---
    stabilized_video = stabilize_video(frames, dy, dtheta)
    stab_h, stab_w = stabilized_video[0].shape[:2]

    # --- PHASE 2: STITCHING ---
    # Calculate canvas width based on raw motion (no stretch factor)
    total_dx = np.sum(np.abs(dx))
    canvas_w = int(total_dx) + w + 500

    panorama = np.zeros((stab_h, canvas_w, 3), dtype=np.uint8)
    current_x = 0

    # Ideally use the center column to minimize perspective distortion
    prespective_col = np.clip(k, 0, w - 1)

    print(f"Phase 2: Stitching strips from column {k}...")

    for i in range(len(frames) - 1):
        frame = stabilized_video[i]

        # 1. Get raw movement
        move_x = abs(dx[i + 1])

        # 2. SKIP DUPLICATE FRAMES
        # Even with perfect LK, the video still has duplicate frames where motion is 0.
        if move_x < 0.5:
            continue

        # 3. Calculate Strip Width
        # We rely strictly on the integer rounded value of the detected motion
        strip_width = int(round(move_x))

        if strip_width <= 0:
            continue

        # 4. Define Boundaries
        col_end_target = prespective_col + strip_width

        # Safety check: ensure we don't read past the image edge
        col_end_actual = min(col_end_target, stab_w)

        actual_width = col_end_actual - prespective_col

        if actual_width <= 0:
            continue

        # 5. Paste
        if current_x + actual_width > canvas_w:
            break

        strip = frame[:, prespective_col: col_end_actual, :]
        panorama[:, current_x: current_x + actual_width, :] = strip

        # Advance cursor by the actual amount we managed to paste
        current_x += actual_width

    # --- FINAL CROP ---
    print("Cropping...")
    mask = panorama.max(axis=2) > 0
    rows, cols = np.where(mask)
    if len(rows) > 0:
        return panorama[np.min(rows): np.max(rows) + 1, np.min(cols): np.max(cols) + 1, :]
    return panorama


def create_panorama(video_path:str, k:int ,shifts_path:str=None) -> np.ndarray:
    video = read_video(video_path)[:-1]
    if shifts_path:
        data = np.load(shifts_path)
        dx, dy, dtheta = data['dx'], data['dy'], data['dtheta']
    else:
        dx, dy, dtheta = get_video_shifts(video)
    return strip_stitching(video, dx, dy, dtheta, k)

def main_save_shifts():
    paths = os.listdir("Exercise Inputs")
    for path in paths:
        if f"{path}_shifts.npz" not in os.listdir():
            video_path = os.path.join("Exercise Inputs", path)
            dx, dy, dtheta = get_video_shifts(read_video(video_path))
            np.savez(f"{path}_shifts.npz", dx=dx, dy=dy, dtheta=dtheta)

def main_save_shift(names):
    for name in names:
        video_path = os.path.join("Exercise Inputs", name)
        dx, dy, dtheta = get_video_shifts(read_video(video_path))
        np.savez(f"{name}_shifts.npz", dx=dx, dy=dy, dtheta=dtheta)



def main_panorams():
    ks = [50, 100, 150, 200, 250, 300, 350, 400]
    for file_name in os.listdir("Exercise Inputs"):
        for k in ks:
            panorama = create_panorama("Exercise Inputs" + os.sep + file_name, k, f"{file_name}_shifts.npz")
            show_image(panorama)


def main_panorama(file_names: List[str] = None):
    for file_name in file_names:
        for k in ks:
            panorama = create_panorama("Exercise Inputs" + os.sep + file_name, k, f"{file_name}_shifts.npz")
            show_image(panorama)
kessaria = "Kessaria.mp4"
boat = "boat.mp4"
garden = "Garden.mp4"
house = "House.mp4"
ks = [50, 100, 150, 200, 250, 300, 350, 400]
if __name__ == "__main__":
    panorama = create_panorama("Exercise Inputs" + os.sep + kessaria, 300, f"{kessaria}_cv.npz")
    show_image(panorama)

