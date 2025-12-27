from ex4 import *

video_path = "v.mp4"
video = read_video(video_path)

# The '...' keeps all frames, heights, and widths
# The '::-1' reverses the color channels (BGR -> RGB)
fixed_video = video[:, :, :-100 , ::-1]

save_video(fixed_video, "iguazu.mp4")


