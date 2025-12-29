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
            cv2_dx.append(0);
            cv2_dy.append(0);
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


def compare_shifts(path, data_path):
    # --- 1. Load Data ---
    data = np.load(data_path)
    if len(data['dx']) > len(read_video(path)):
        my_dx = data['dx'][1:]
    else:
        my_dx = data['dx']

    cv2_dx, cv2_dy = get_shifts_opencv_robust(path)

    min_len = min(len(my_dx), len(cv2_dx))
    my_dx = my_dx[:min_len]
    cv2_dx = cv2_dx[:min_len]

    # --- 2. Print Detailed Table (First 10 frames) ---
    print(f"\n{'Frame':<6} | {'My LK':<10} | {'OpenCV':<10} | {'Diff'}")
    print("-" * 45)
    for i in range(min_len):
        diff = abs(my_dx[i] - cv2_dx[i])
        match = "✅" if diff < 1.5 else "❌"
        print(f"{i:<6} | {my_dx[i]:<10.2f} | {cv2_dx[i]:<10.2f} | {diff:<5.2f} {match}")
    print("...")

    # --- 3. CALCULATE METRICS ---
    # RMSE: Lower is better (0 is perfect)
    rmse = np.sqrt(np.mean((my_dx - cv2_dx) ** 2))

    # Correlation: Higher is better (1.0 is perfect)
    if np.std(my_dx) > 1e-5 and np.std(cv2_dx) > 1e-5:
        correlation = np.corrcoef(my_dx, cv2_dx)[0, 1]
    else:
        correlation = 0.0

    # Success Rate (Tolerance: 2 pixels)
    success_count = np.sum(np.abs(my_dx - cv2_dx) < 2.0)
    success_rate = (success_count / min_len) * 100

    # --- 4. PRINT FINAL SCORE ---
    print("\n" + "=" * 60)
    print("📊  COMPARISON QUALITY METRICS")
    print("=" * 60)

    print(f"1. Success Rate (Tolerance < 2px):  {success_rate:.1f}%")
    print(f"   (Percentage of frames considered accurate)")

    print(f"2. Pearson Correlation:             {correlation:.4f}  (Target: > 0.9)")
    print(f"   (Measures trend similarity. 1.0 = Perfect trend match)")

    print(f"3. RMSE (Root Mean Square Error):   {rmse:.3f} px   (Target: < 2.0)")
    print(f"   (Average error magnitude. Lower is better)")

    print("-" * 60)

    # Final Verdict
    if correlation > 0.9 and rmse < 2.0:
        print("🏆  FINAL VERDICT: EXCELLENT Match (High accuracy)")
    elif correlation > 0.7:
        print("✅  FINAL VERDICT: GOOD Match (Valid algorithm, slight noise)")
    else:
        print("⚠️  FINAL VERDICT: POOR Match (Check for bugs or 'Aperture Problem')")
    print("=" * 60)

video = iguazu
# compare_shifts(fr"Exercise Inputs/{video}", fr"shifts/{video}_shifts.npz")
a = np.zeros(15)
b = np.array([1, 2])
a[:] = b
print(a)