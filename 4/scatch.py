from ex4 import *


def get_shifts_opencv_robust(video_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates "Ground Truth" shifts using OpenCV's built-in
    Sparse Lucas-Kanade with outlier rejection (Median).
    """
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        return np.array([]), np.array([])

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Feature params for Good Features to Track (Strong corners only)
    feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)

    # LK params
    lk_params = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Detect initial points (ignore sky automatically)
    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    dx_list = []
    dy_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate Optical Flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        if p1 is not None and st is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # Calculate movement for all valid points
            deltas = good_new - good_old

            # ROBUSTNESS: Take the Median to ignore outliers (moving objects)
            if len(deltas) > 0:
                dx = np.median(deltas[:, 0])
                dy = np.median(deltas[:, 1])
            else:
                dx, dy = 0.0, 0.0

            dx_list.append(dx)
            dy_list.append(dy)

            # Update points for next frame
            prev_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

            # Re-detect points if too few remain
            if len(p0) < 10:
                p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
        else:
            dx_list.append(0.0)
            dy_list.append(0.0)
            prev_gray = frame_gray.copy()
            p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    cap.release()

    # Pad with 0 at the start to match your array length style (dx[0] is usually 0)
    return np.array([0] + dx_list), np.array([0] + dy_list)


def compare_shifts(path):
    print(f"--- Comparing: {path} ---")

    # --- 1. Load Data ---
    # Unpack your custom results
    print("Calculating MY shifts...")
    video = mp.read_video(path)
    # Note: get_video_shifts returns (dx, dy, theta). We take index 0 (dx) and 1 (dy).
    my_results = get_video_shifts(video)
    my_dx = my_results[0]
    my_dy = my_results[1]

    # Calculate Ground Truth
    print("Calculating OpenCV shifts...")
    cv2_dx, cv2_dy = get_shifts_opencv_robust(path)

    # Align lengths
    min_len = min(len(my_dx), len(cv2_dx))
    # Ignore the very first frame (usually 0) and very last frame (often artifacts)
    # Testing frames 1 to N-1 is usually most accurate
    compare_slice = slice(1, min_len - 1)

    my_dx_cut = my_dx[compare_slice]
    cv2_dx_cut = cv2_dx[compare_slice]

    # --- 2. Print Detailed Table (First 10 frames) ---
    print(f"\n{'Frame':<6} | {'My LK':<10} | {'OpenCV':<10} | {'Diff'}")
    print("-" * 45)
    for i in range(min(15, len(my_dx_cut))):
        diff = abs(my_dx_cut[i] - cv2_dx_cut[i])
        match = "✅" if diff < 1.0 else "❌"  # Stricter 1.0 pixel tolerance
        print(f"{i + 1:<6} | {my_dx_cut[i]:<10.2f} | {cv2_dx_cut[i]:<10.2f} | {diff:<5.2f} {match}")
    print("...")

    # --- 3. CALCULATE METRICS ---
    # RMSE
    rmse = np.sqrt(np.mean((my_dx_cut - cv2_dx_cut) ** 2))

    # Correlation
    if np.std(my_dx_cut) > 1e-5 and np.std(cv2_dx_cut) > 1e-5:
        correlation = np.corrcoef(my_dx_cut, cv2_dx_cut)[0, 1]
    else:
        correlation = 0.0

    # Success Rate (Tolerance: 1.0 pixel)
    success_count = np.sum(np.abs(my_dx_cut - cv2_dx_cut) < 1.0)
    success_rate = (success_count / len(my_dx_cut)) * 100

    # --- 4. PRINT FINAL SCORE ---
    print("\n" + "=" * 60)
    print("📊  COMPARISON QUALITY METRICS")
    print("=" * 60)
    print(f"Video: {path}")

    print(f"1. Success Rate (Diff < 1px):       {success_rate:.1f}%")

    print(f"2. Pearson Correlation:             {correlation:.4f}  (Target: > 0.9)")

    print(f"3. RMSE (Root Mean Square Error):   {rmse:.3f} px   (Target: < 1.0)")

    print("-" * 60)

    # Final Verdict
    if correlation > 0.95 and rmse < 1.0:
        print("🏆  FINAL VERDICT: EXCELLENT Match (Professional Grade)")
    elif correlation > 0.85 and rmse < 2.0:
        print("✅  FINAL VERDICT: GOOD Match (Valid for Assignment)")
    elif correlation > 0.6:
        print("⚠️  FINAL VERDICT: OKAY Match (Roughly follows trend)")
    else:
        print("❌  FINAL VERDICT: POOR Match (Algorithm divergent)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Ensure paths are correct
    boat_path = os.path.join("Exercise Inputs", "boat.mp4")
    kessaria_path = os.path.join("Exercise Inputs", "Kessaria.mp4")  # The hard one (sky)

    # Compare
    compare_shifts(boat_path)
    compare_shifts(kessaria_path)