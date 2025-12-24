from ex4 import *

video = mp.read_video('iguazu_video.mp4')
reversed_video = video[::-1]
save_video(reversed_video, 'reversed_iguazu.mp4')
