from typing import Tuple, Any, List
from matplotlib import pyplot as plt
from mediapy import read_video, show_video
from scipy.signal import convolve2d
from square_video import *

FILTER_SIZE = 3
PYRAMID_LEVELS = 7
ITERATIONS_PER_LEVEL = 15

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

def lk_prep_image(im):
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.dtype != np.float32:
        im = im.astype(np.float32)
    # Normalize to 0-1 if in 0-255 range for stability
    if im.max() > 1.0:
        im /= 255.0
    return im


def lucas_kanade(frame1: np.ndarray, frame2: np.ndarray) -> Tuple[float, float, float]:
    """
    Simultaneous Pyramidal Lucas-Kanade for Translation (u, v) and Rotation (theta).
    Solves a 3x3 system [du, dv, dtheta] at each iteration.
    """
    # 1. Prep Images
    I1 = lk_prep_image(frame1)
    I2 = lk_prep_image(frame2)

    # 2. Light Blur
    I1 = cv2.GaussianBlur(I1, (3, 3), 0)
    I2 = cv2.GaussianBlur(I2, (3, 3), 0)

    # 3. Build Pyramids
    pyr1 = build_single_channel_gaussian_pyramid(I1, PYRAMID_LEVELS, FILTER_SIZE)
    pyr2 = build_single_channel_gaussian_pyramid(I2, PYRAMID_LEVELS, FILTER_SIZE)

    u, v, theta = 0.0, 0.0, 0.0

    # 4. Coarse-to-Fine Loop
    for level in range(len(pyr1) - 1, -1, -1):
        u *= 2
        v *= 2

        im1_lvl = pyr1[level]
        im2_lvl = pyr2[level]
        h, w = im1_lvl.shape
        cy, cx = h / 2.0, w / 2.0

        y_grid, x_grid = np.mgrid[0:h, 0:w]
        x_grid = x_grid.astype(np.float32) - cx
        y_grid = y_grid.astype(np.float32) - cy

        for _ in range(ITERATIONS_PER_LEVEL):
            # --- A. Warp I1 towards I2 ---
            # FIX 1: Subtract u and v to shift image in the correct direction
            M = cv2.getRotationMatrix2D((cx, cy), -theta, 1.0)
            M[0, 2] -= u
            M[1, 2] -= v

            im1_warp = cv2.warpAffine(
                im1_lvl, M, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT
            )

            # --- B. Compute Error and Gradients ---
            It = im2_lvl - im1_warp

            Ix = cv2.Sobel(im1_warp, cv2.CV_64F, 1, 0, ksize=3)
            Iy = cv2.Sobel(im1_warp, cv2.CV_64F, 0, 1, ksize=3)

            # FIX 2: Swap terms for Y-Down coordinate system
            # J_theta = y*Ix - x*Iy
            J_theta = (y_grid * Ix - x_grid * Iy) * (np.pi / 180.0)

            # --- D. Build 3x3 System ---
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

            b = np.array([
                np.dot(Ix_f, It_f),
                np.dot(Iy_f, It_f),
                np.dot(Jth_f, It_f)
            ])

            # --- E. Solve and Update ---
            try:
                delta = np.linalg.pinv(A) @ b
                du, dv, dtheta = delta

                u += du
                v += dv
                theta += dtheta

                if abs(du) < 1e-4 and abs(dv) < 1e-4 and abs(dtheta) < 1e-4:
                    break
            except np.linalg.LinAlgError:
                break

    return -u, -v, -theta

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


def stabilize_video(frames: np.ndarray, dy: np.ndarray, dtheta: np.ndarray) -> np.ndarray:
    """
    Phase 1: Aligns all frames to the MIDDLE frame's coordinate system.
    """
    h, w = frames[0].shape[:2]
    num_frames = len(frames)
    mid_idx = num_frames // 2

    # 1. Calculate Cumulative Shifts (relative to frame 0)
    # cumsum gives us the position of every frame relative to the start
    cumulative_dy = np.cumsum(dy)
    cumulative_theta = np.cumsum(dtheta)

    # 2. Re-center everything to the Middle Frame
    # By subtracting the middle frame's value, the middle frame becomes 0 (the anchor),
    # frames before it become negative, and frames after it become positive.
    abs_dy = cumulative_dy - cumulative_dy[mid_idx]
    abs_theta = cumulative_theta - cumulative_theta[mid_idx]

    # 3. Calculate Canvas Height
    # We need room for the max deviation in both directions (up and down)
    max_y_deviation = int(np.max(np.abs(abs_dy)))
    aligned_h = h + (2 * max_y_deviation) + 200  # Buffer

    # The middle frame will sit exactly at the center of this new canvas
    canvas_center_y = aligned_h // 2

    original_center_x, original_center_y = w // 2, h // 2

    stabilized_frames = []

    for i in range(num_frames):
        # Current deviations relative to the middle frame
        current_angle = abs_theta[i]
        current_dy_shift = abs_dy[i]

        # --- Build Transformation Matrix ---

        # 1. Rotation
        # We rotate by -current_angle to "undo" the rotation relative to the center frame
        M = cv2.getRotationMatrix2D((original_center_x, original_center_y), -current_angle, 1.0)

        # 2. Vertical Translation
        # We want the image center (h//2) to land at: CanvasCenter - Deviation
        # If frame i is "higher" than middle (positive dy), we must shift it DOWN to align.
        # Note: In images, +y is down. If camera moved DOWN (+dy), image moved UP.
        # To stabilize, we usually subtract the motion.

        # Calculate target y position for this frame's center
        target_y = canvas_center_y - current_dy_shift

        # The shift needed = Target - Original
        M[1, 2] += (target_y - original_center_y)

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

    return np.array(stabilized_frames)


def strip_stitching(frames: np.ndarray, dx: np.ndarray, dy: np.ndarray, dtheta: np.ndarray, k: int) -> np.ndarray:
    """
    Phase 2: Stitching with optional stretch factor for difficult videos (e.g. Waterfalls, Low FPS).
    """
    h, w = frames[0].shape[:2]

    # --- PHASE 1: ALIGNMENT ---
    stabilized_video = stabilize_video(frames, dy, dtheta)
    stab_h, stab_w = stabilized_video[0].shape[:2]

    # --- PHASE 2: STITCHING ---
    # Apply stretch factor to total width calculation
    # We add 'w' again to ensure we have enough buffer
    total_dx = np.sum(np.abs(dx))
    canvas_w = int(total_dx) + w + 1000

    panorama = np.zeros((stab_h, canvas_w, 3), dtype=np.uint8)
    current_x = 0

    prespective_col = np.clip(k, 0, w - 1)

    for i in range(len(frames) - 1):
        frame = stabilized_video[i]
        move_x = abs(dx[i + 1])

        if move_x < 0.1:  # Threshold for "no motion"
            continue

        # Apply stretch factor to the strip width
        target_width = int(round(move_x))

        if target_width <= 0:
            continue

        col_end_target = prespective_col + target_width
        col_end_actual = min(col_end_target, stab_w)
        actual_width = col_end_actual - prespective_col

        if actual_width <= 0:
            continue

        if current_x + actual_width > canvas_w:
            break

        strip = frame[:, prespective_col: col_end_actual, :]
        panorama[:, current_x: current_x + actual_width, :] = strip

        current_x += actual_width

    mask = panorama.max(axis=2) > 0
    rows, cols = np.where(mask)
    if len(rows) > 0:
        return panorama[np.min(rows): np.max(rows) + 1, np.min(cols): np.max(cols) + 1, :]
    return panorama


def create_panorama(video_path:str, k:int ,shifts_path:str=None) -> np.ndarray:
    video = read_video(video_path)
    if shifts_path:
        data = np.load(shifts_path)
        dx, dy, dtheta = data['dx'], data['dy'], data['dtheta']
    else:
        dx, dy, dtheta = get_video_shifts(video)
    return strip_stitching(video, dx, dy, dtheta, k)


def create_video_animation(video_path, ks, shifts_path=None) -> np.ndarray:
    panoramas = []
    # 1. Generate all panoramas first
    for k in ks:
        print("creating panorama for k={}".format(k))
        pano = create_panorama("Exercise Inputs" + os.sep + video_path, k, shifts_path)
        panoramas.append(pano)

    # 2. Find the minimum common dimensions
    min_h = min(p.shape[0] for p in panoramas)
    min_w = min(p.shape[1] for p in panoramas)
    cropped_panos = []

    # 3. Center-Crop all images to the minimum size
    for p in panoramas:
        h, w = p.shape[:2]

        # Calculate crop offsets to center the image
        start_y = (h - min_h) // 2
        start_x = (w - min_w) // 2

        # Perform the crop
        crop = p[start_y: start_y + min_h, start_x: start_x + min_w, :]
        cropped_panos.append(crop)

    # 4. Stack into a video array (Frames, Height, Width, Channels)
    video_array = np.stack(cropped_panos, axis=0)

    # 5. Show/Save
    # 2 FPS is usually good to see the perspective shift clearly
    show_video(video_array, fps=2)
    return video_array

def main_save_shifts(names):
    for name in names:
        video_path = os.path.join("Exercise Inputs", name)
        dx, dy, dtheta = get_video_shifts(read_video(video_path))
        np.savez(fr"shifts/{name}_shifts.npz", dx=dx, dy=dy, dtheta=dtheta)

def main_panorama(file_names: List[str] = None):
    ks = [_ for _ in range(0, 420, 50)]
    for file_name in file_names:
        for k in ks:
            panorama = create_panorama("Exercise Inputs" + os.sep + file_name, k, fr"shifts/{file_name}_shifts.npz")
            show_image(panorama)

def main_create_video(file_name: str, ks: List[float]) -> None:

    result_video = create_video_animation(file_name, ks, f"shifts/{file_name}_shifts.npz")
    save_video(result_video, f"video outputs/{file_name}.mp4")


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
boat_ks = [_ for _ in range(30, 420, 3)]
iguazu_ks = [_ for _ in range(160, 490, 10)]
my_videos_ks = [_ for _ in range(10, 420, 5)]
if __name__ == "__main__":
    main_panorama([kessaria])
