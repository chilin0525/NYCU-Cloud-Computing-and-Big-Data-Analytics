train_dir_path = "train"
test_dir_path = "test"

from moviepy.editor import VideoFileClip, vfx
import os
train_dirs = os.listdir(train_dir_path)

cnt = 0
for class_dir in train_dirs:
    class_dir_files = os.listdir(f"{train_dir_path}/{class_dir}")
    for i in class_dir_files:
        clip = VideoFileClip(f"{train_dir_path}/{class_dir}/{i}")
        # reversed_clip = clip.rotate(45)
        reversed_clip = clip.fx(vfx.mirror_x)
        reversed_clip.write_videofile(f"{train_dir_path}/{class_dir}/rotate_45_{i}")
    cnt += 1