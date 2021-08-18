import os
import subprocess

base_dir = None #'./MTL-AQA/new/new_total_frames'
save_dir = None #'./MTL-AQA/new/new_total_frames_256s'
if base_dir is None:
    raise RuntimeError('please choose the base dir and save dir')
video_list = os.listdir(base_dir)
for video in video_list:
    save_path = os.path.join(save_dir,video)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    frame_list = os.listdir(os.path.join(base_dir,video))
    for frame in frame_list:
        subprocess.call('ffmpeg -i %s -vf scale=-1:256 %s' % (os.path.join(base_dir, video, frame), \
                                    os.path.join(save_dir, video, frame)), shell=True )

