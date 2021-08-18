### Download MTL-AQA raw videos
```
# see https://github.com/ytdl-org/youtube-dl/
youtube-dl -f best <video_url> --default-search "ytsearch" --verbose -o <dst_path>
# use a proxy if you cant visit youtube directly
```

### get frames
```
ffmpeg -i <video> -f image2 -qscale:v 2 ./new/new_total_frames/<video_id>/%08d.jpg
```

### [optional] Resize the frames to save the cost
```
python ./MTL_AQA/MTL_tool/resize_frames.py
```
