import numpy as np
import mediapy as mp

RED_TO_GREY = 0.299
GREEN_TO_GREY = 0.587
BLUE_TO_GREY = 0.114
def get_grayscale_video(video_path: str) -> np.ndarray:
    """
    Convert an RGB video to grayscale using the formula:
    Gray = 0.299*R + 0.587*G + 0.114*B

    :param video_path: path to video file
    :return: a numpy array of shape (num_frames, H, W) representing the grayscale video
    """
    rgb_video = mp.read_video(video_path)
    grayscale_video = (rgb_video[:, :, :, 0] * RED_TO_GREY + rgb_video[:, :, :, 1] *
                       GREEN_TO_GREY + rgb_video[:, :, :, 2] * BLUE_TO_GREY)
    return grayscale_video.astype(np.uint8)

def get_video_cumulative_histograms(video_frame: np.ndarray) -> np.ndarray:
    """
    Compute the cumulative histogram of pixel intensities for each frame in the grayscale video.

    :param video_frame: a numpy array of shape (num_frames, H, W) representing the grayscale video
    :return: a numpy array of shape (num_frames, 256) representing the histogram of each frame
    """
    num_frames = video_frame.shape[0]
    histogram = np.zeros((num_frames, 256), dtype=int)

    for i in range(num_frames):
        frame = video_frame[i]
        hist, _ = np.histogram(frame, bins=256, range=(0, 256))
        histogram[i] = hist

    return np.cumsum(histogram, axis=1)

# def show_frame(video: np.ndarray, frame: np.ndarray) -> None:
#     """
#     Display a single frame from the video.
#
#     :param video: a numpy array of shape (num_frames, H, W, 3) representing the RGB video
#     :param frame: an integer representing the frame number to display
#     """
#     plt.axis('off')
#     plt.imshow(video[frame])
#     plt.show()

def main(video_path, video_type):
    """
    Main entry point for ex1
    :param video_path: path to video file
    :param video_type: category of the video (either 1 or 2)
    :return: a tuple of integers representing the frame number for which the scene cut was detected (i.e. the last frame index of the first scene and the first frame index of the second scene)
    """
    grayscale_video = get_grayscale_video(video_path)
    cumulative_histogram = get_video_cumulative_histograms(grayscale_video)
    frame_diffs = np.sum(np.abs(np.diff(cumulative_histogram, axis=0)), axis=1)
    cut_frame = np.argmax(frame_diffs)
    return int(cut_frame), int(cut_frame + 1)
