import numpy as np

from ex4 import *


def get_shifts_opencv_robust(video_path):
    """
    Calculates shifts using OpenCV's optimized C++ functions:
    1. cv2.goodFeaturesToTrack (Finds strong corners)
    2. cv2.calcOpticalFlowPyrLK (Tracks them)
    3. cv2.estimateAffinePartial2D (Finds global rotation/translation using RANSAC)
    """
    video = read_video(video_path)
    num_frames = len(video)

    cv2_dx = []
    cv2_dy = []
    cv2_theta = []

    print("Computing Ground Truth using OpenCV...")

    prev_gray = cv2.cvtColor(video[0], cv2.COLOR_RGB2GRAY)

    for i in range(1, num_frames):
        curr_gray = cv2.cvtColor(video[i], cv2.COLOR_RGB2GRAY)

        # 1. Find good features to track in the previous frame
        p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)

        if p0 is None:
            cv2_dx.append(0);
            cv2_dy.append(0);
            cv2_theta.append(0)
            continue

        # 2. Run Lucas Kanade (OpenCV Built-in)
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None)

        # Select good points (status == 1)
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # 3. Estimate Transformation Matrix (RANSAC handles outliers!)
        # estimateAffinePartial2D finds best [Rotation, Translation, Scale]
        # that maps old points to new points.
        M, inliers = cv2.estimateAffinePartial2D(good_old, good_new)

        if M is None:
            cv2_dx.append(0)
            cv2_dy.append(0)
            cv2_theta.append(0)
        else:
            # M is [[cos, -sin, tx], [sin, cos, ty]]
            tx = M[0, 2]
            ty = M[1, 2]

            # Extract angle from rotation matrix components
            # theta = arctan2(sin, cos)
            theta_rad = np.arctan2(M[1, 0], M[0, 0])
            theta_deg = np.degrees(theta_rad)

            # Note: OpenCV calculates transform FROM prev TO curr.
            # Your code seems to calculate how much to shift curr TO match prev.
            # We usually invert the sign to match "shift" logic, but let's check magnitude first.
            cv2_dx.append(tx)
            cv2_dy.append(ty)
            cv2_theta.append(theta_deg)

        prev_gray = curr_gray

    return np.array(cv2_dx), np.array(cv2_dy)


# --- MAIN COMPARISON BLOCK ---
# 1. Load your existing calculated shifts (assuming you saved them)
# If not, run your 'get_video_shifts' function here
name = kessaria
path = f"Exercise Inputs/{name}"
data = np.load(f"{name}_shifts.npz")
my_dx = data['dx'][1:]  # Skip the first 0 if your array includes it

# 2. Get OpenCV shifts
cv2_dx, cv2_dy = get_shifts_opencv_robust(path)
dtheta = np.zeros_like(cv2_dx)
np.savez(f"{kessaria}_cv.npz", dx=cv2_dx, dy=cv2_dy, dtheta=dtheta)
