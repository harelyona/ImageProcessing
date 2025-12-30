import glob
from typing import List, Tuple
from matplotlib import pyplot as plt
import mediapy as mp
import cv2
import numpy as np
import os
from PIL import Image

BLUR_FILTER_SIZE = 5
PYRAMID_LEVELS = 5
ITERATIONS_PER_LEVEL = 5


def show_image(img, save_path=None):
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    if save_path:
        plt.imsave(save_path, img, cmap='gray')
    plt.show()


# --- Pyramid Helpers ---

def expand(image):
    return cv2.pyrUp(image)


def build_single_channel_gaussian_pyramid(im, max_levels):
    pyr = [im]
    current_im = im
    for _ in range(max_levels - 1):
        downsampled_im = cv2.pyrDown(current_im)

        # Stop if height or width goes below 30
        if downsampled_im.shape[0] < 30 or downsampled_im.shape[1] < 30:
            break

        pyr.append(downsampled_im)
        current_im = downsampled_im
    return pyr


def build_single_channel_laplacian_pyramid(image, max_levels):
    gaussian_pyr = build_single_channel_gaussian_pyramid(image, max_levels)
    laplacian_pyr = []
    for i in range(len(gaussian_pyr) - 1):
        g_current = gaussian_pyr[i]
        g_next = gaussian_pyr[i + 1]
        expanded_next = expand(g_next)
        if expanded_next.shape != g_current.shape:
            expanded_next = expanded_next[:g_current.shape[0], :g_current.shape[1]]
        laplacian_pyr.append(g_current - expanded_next)
    laplacian_pyr.append(gaussian_pyr[-1])
    return laplacian_pyr


# --- Lucas-Kanade Logic ---

def lk_prep_image(im):
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.dtype != np.float32:
        im = im.astype(np.float32)
    if im.max() > 1.0:
        im /= 255.0
    return im


def lk_warp(image: np.ndarray, u: float, v: float, theta: float) -> np.ndarray:
    h, w = image.shape
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), -theta, 1.0)
    M[0, 2] -= u
    M[1, 2] -= v
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def lk_gradients(warped_im: np.ndarray, target_im: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    It = target_im - warped_im
    Ix = cv2.Sobel(warped_im, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(warped_im, cv2.CV_64F, 0, 1, ksize=3)
    return Ix, Iy, It


def lk_solve(Ix: np.ndarray, Iy: np.ndarray, It: np.ndarray,
             x_grid: np.ndarray, y_grid: np.ndarray) -> Tuple[float, float, float]:
    h, w = Ix.shape
    J_theta = (y_grid * Ix - x_grid * Iy) * (np.pi / 180.0)
    margin_y = int(h * 0.15)
    margin_x = int(w * 0.15)

    if margin_y > 0 and margin_x > 0 and (h - 2 * margin_y) > 2 and (w - 2 * margin_x) > 2:
        sl_y = slice(margin_y, -margin_y)
        sl_x = slice(margin_x, -margin_x)
    else:
        sl_y = slice(None)
        sl_x = slice(None)

    Ix_f = Ix[sl_y, sl_x].flatten()
    Iy_f = Iy[sl_y, sl_x].flatten()
    Jth_f = J_theta[sl_y, sl_x].flatten()
    It_f = It[sl_y, sl_x].flatten()

    Ixx = np.dot(Ix_f, Ix_f)
    Iyy = np.dot(Iy_f, Iy_f)
    Itt = np.dot(Jth_f, Jth_f)
    Ixy = np.dot(Ix_f, Iy_f)
    Ixt = np.dot(Ix_f, Jth_f)
    Iyt = np.dot(Iy_f, Jth_f)

    A = np.array([[Ixx, Ixy, Ixt], [Ixy, Iyy, Iyt], [Ixt, Iyt, Itt]])

    # Regularization
    epsilon = 0.5
    A[0, 0] += epsilon
    A[1, 1] += epsilon
    A[2, 2] += epsilon

    b = np.array([np.dot(Ix_f, It_f), np.dot(Iy_f, It_f), np.dot(Jth_f, It_f)])

    try:
        delta = np.linalg.pinv(A) @ b
        return delta[0], delta[1], delta[2]
    except np.linalg.LinAlgError:
        return 0.0, 0.0, 0.0


def lucas_kanade(frame1: np.ndarray, frame2: np.ndarray) -> Tuple[float, float, float]:
    # Define your convergence threshold
    EPSILON = 1e-2
    # Safety limit: Prevents infinite loops if the image has no texture or algorithm diverges
    MAX_ITER = 100

    I1 = lk_prep_image(frame1)
    I2 = lk_prep_image(frame2)

    # Double Blur (As per your code)
    for _ in range(3):
        I1 = cv2.GaussianBlur(I1, (BLUR_FILTER_SIZE, BLUR_FILTER_SIZE), 0)
        I2 = cv2.GaussianBlur(I2, (BLUR_FILTER_SIZE, BLUR_FILTER_SIZE), 0)

    pyr1 = build_single_channel_gaussian_pyramid(I1, PYRAMID_LEVELS)
    pyr2 = build_single_channel_gaussian_pyramid(I2, PYRAMID_LEVELS)

    u, v, theta = 0.0, 0.0, 0.0

    for level in range(len(pyr1) - 1, -1, -1):
        u *= 2
        v *= 2

        im1_lvl = pyr1[level]
        im2_lvl = pyr2[level]
        h, w = im1_lvl.shape
        cy, cx = h / 2.0, w / 2.0

        # Grid construction
        y_grid, x_grid = np.mgrid[0:h, 0:w]
        x_grid = x_grid.astype(np.float32) - cx
        y_grid = y_grid.astype(np.float32) - cy

        # --- CHANGED: While loop for convergence ---
        iteration_count = 0
        while iteration_count < MAX_ITER:
            iteration_count += 1

            im1_warp = lk_warp(im1_lvl, u, v, theta)
            Ix, Iy, It = lk_gradients(im1_warp, im2_lvl)

            # Note: Ensure lk_solve has the negative sign fix!
            du, dv, dtheta = lk_solve(Ix, Iy, It, x_grid, y_grid)

            u += du
            v += dv
            theta += dtheta

            # Convergence Check
            if abs(du) < EPSILON and abs(dv) < EPSILON and abs(dtheta) < EPSILON:
                break
        # -------------------------------------------

    return -u, -v, -theta


def get_video_shifts(video: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_frames = video.shape[0]
    x_shifts = np.zeros(num_frames)
    y_shifts = np.zeros(num_frames)
    th_shifts = np.zeros(num_frames)
    for i in range(num_frames - 1):
        x_shifts[i + 1], y_shifts[i + 1], th_shifts[i + 1] = lucas_kanade(video[i], video[i + 1])
    return x_shifts, y_shifts, th_shifts


def stabilize_video(frames: np.ndarray, dy: np.ndarray, dtheta: np.ndarray) -> np.ndarray:
    """
    Stabilizes the video once.
    """
    h, w = frames[0].shape[:2]
    num_frames = len(frames)
    mid_idx = num_frames // 2

    cumulative_dy = np.cumsum(dy)
    cumulative_theta = np.cumsum(dtheta)

    abs_dy = cumulative_dy - cumulative_dy[mid_idx]
    abs_theta = cumulative_theta - cumulative_theta[mid_idx]

    max_y_deviation = int(np.max(np.abs(abs_dy)))
    aligned_h = h + (2 * max_y_deviation) + 200
    canvas_center_y = aligned_h // 2
    original_center_x, original_center_y = w // 2, h // 2

    stabilized_frames = []

    for i in range(num_frames):
        current_angle = abs_theta[i]
        current_dy_shift = abs_dy[i]

        M = cv2.getRotationMatrix2D((original_center_x, original_center_y), -current_angle, 1.0)
        target_y = canvas_center_y - current_dy_shift
        M[1, 2] += (target_y - original_center_y)

        warped_frame = cv2.warpAffine(
            frames[i], M, (w, aligned_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        stabilized_frames.append(warped_frame)

    return np.array(stabilized_frames)


def stitch_stabilized_video(stabilized_frames: np.ndarray, dx: np.ndarray, k: int) -> np.ndarray:
    """
    Stitches a panorama from ALREADY stabilized frames.
    Fast!
    """
    stab_h, stab_w = stabilized_frames[0].shape[:2]

    # Note: stab_w is equal to original w (rotation preserves width in our warp logic)
    # So we use stab_w for clipping k.

    total_dx = np.sum(np.abs(dx))
    canvas_w = int(total_dx) + stab_w + 1000
    panorama = np.zeros((stab_h, canvas_w, 3), dtype=np.uint8)

    current_x = 0
    prespective_col = np.clip(k, 0, stab_w - 1)

    for i in range(len(stabilized_frames) - 1):
        frame = stabilized_frames[i]
        move_x = abs(dx[i + 1])

        if move_x < 0.1: continue

        target_width = int(round(move_x))
        if target_width <= 0: continue

        col_end_target = prespective_col + target_width
        col_end_actual = min(col_end_target, stab_w)
        actual_width = col_end_actual - prespective_col

        if actual_width <= 0: continue
        if current_x + actual_width > canvas_w: break

        strip = frame[:, prespective_col: col_end_actual, :]
        panorama[:, current_x: current_x + actual_width, :] = strip
        current_x += actual_width

    mask = panorama.max(axis=2) > 0
    rows, cols = np.where(mask)
    if len(rows) > 0:
        return panorama[np.min(rows): np.max(rows) + 1, np.min(cols): np.max(cols) + 1, :]
    return panorama


def create_video_animation(video: np.ndarray, ks: List[int]) -> np.ndarray:
    """
    Optimized Animation Generator with Timing.
    """
    dx, dy, dtheta = get_video_shifts(video)
    stabilized_video = stabilize_video(video, dy, dtheta)
    panoramas = []
    for k in ks:
        pano = stitch_stabilized_video(stabilized_video, dx, k)
        panoramas.append(pano)
    if not panoramas:
        return np.array([])

    min_h = min(p.shape[0] for p in panoramas)
    min_w = min(p.shape[1] for p in panoramas)
    cropped_panos = []

    for p in panoramas:
        h, w = p.shape[:2]
        start_y = (h - min_h) // 2
        start_x = (w - min_w) // 2
        crop = p[start_y: start_y + min_h, start_x: start_x + min_w, :]
        cropped_panos.append(crop)

    video_array = np.stack(cropped_panos, axis=0)
    return video_array


def save_video(video: np.ndarray, filename: str) -> None:
    if len(video) == 0: return
    height, width, _ = video[0].shape
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 2, (width, height))
    for frame in video:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()


def main_create_video(file_names: List[str], ks: List[int]) -> None:
    for file_name in file_names:
        video = mp.read_video("Exercise Inputs/" + file_name)
        result_video = create_video_animation(video, ks)
        save_video(result_video, f"video outputs/{file_name}")


def generate_panorama(input_frames_path, n_out_frames):
    """
    Main entry point for ex4
    :param input_frames_path : path to a dir with input video frames.
    :param n_out_frames: number of generated panorama frames
    :return: A list of generated panorama frames (of size n_out_frames),
    each list item should be a PIL image of a generated panorama.
    """
    # 1. Load Frames
    search_path = os.path.join(input_frames_path, "*.jpg")
    frame_files = sorted(glob.glob(search_path))

    frames = []
    for f in frame_files:
        img_bgr = cv2.imread(f)
        if img_bgr is not None:
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            frames.append(img_rgb)
    frames = np.array(frames)

    # 2. Define K values (Viewpoints)
    h, w = frames[0].shape[:2]

    # We define a margin to avoid the empty black borders from stabilization.
    # 5% of the width is usually a safe safe-zone.
    margin = int(w * 0.1)

    k_start = margin
    k_end = w - margin

    # Generate exactly n_out_frames integers evenly spaced between start and end
    ks = np.linspace(k_start, k_end, n_out_frames).astype(int).tolist()

    # 3. Create Animation (Using your existing function, assumed to accept 'frames')
    # Returns numpy array of shape (N, H, W, 3)
    video_array = create_video_animation(frames, ks)

    # 4. Convert to List of PIL Images
    pil_panoramas = [Image.fromarray(frame) for frame in video_array]
    return pil_panoramas


kessaria = "Kessaria.mp4"
boat = "boat.mp4"
garden = "Garden.mp4"
house = "House.mp4"
iguazu = "Iguazu.mp4"
shinkansen = "Shinkansen.mp4"
trees = "Trees.mp4"
my_video = "MyVideoNormal.mp4"
my_video_zoom = "MyVideoZoom.mp4"
iguazu_video_path = f"Exercise Inputs/{iguazu}"
boat_data_path = "shifts/boat.mp4_shifts.npz"
all_videos = [kessaria, boat, garden, house, iguazu, shinkansen, trees, my_video, my_video_zoom]
boat_ks = [_ for _ in range(30, 420, 20)]
iguazu_ks = [_ for _ in range(160, 490, 10)]
my_videos_ks = [_ for _ in range(10, 420, 5)]


#main_create_video([kessaria], boat_ks)
