
# extract frames
ffmpeg -i /opt/data2/yuxumin/MTL-AQA/MTL_videos/6.mp4 -f image2 -qscale:v 2 /opt/data2/yuxumin/MTL-AQA/new/new_total_frames/6/%08d.jpg
# count frame number
# ffprobe <input> -select_streams v -show_entries stream=nb_frames -of default=nk=1:nw=1 -v quiet