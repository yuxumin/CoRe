import subprocess

video_list = {
        1:'https://www.youtube.com/watch?v=cYkUl8MrXgA',
        2:'https://www.youtube.com/watch?v=V5parOfpuEQ',
        3:'https://www.youtube.com/watch?v=Y4ARBzok9aU',
        4:'https://www.youtube.com/watch?v=ZAmJsIcfFOA',
        5:'https://www.youtube.com/watch?v=D6zILEKIJbk',
        6:'https://www.youtube.com/watch?v=QhXToslnPvA',
        7:'https://www.youtube.com/watch?v=sexZ6VnZ9yc',
        9:'https://www.youtube.com/watch?v=4_0xwGGMvEM',
        10:'https://www.youtube.com/watch?v=Ain8HstVu7I',
        13:'https://www.youtube.com/watch?v=RWNrARSbRCY',
        14:'https://www.youtube.com/watch?v=bis2HyvgBh4',
        17:'https://www.youtube.com/watch?v=_tigfCJFLZg',
        18:'https://www.youtube.com/watch?v=Bb0ZiYVNtDs',
        22:'https://www.youtube.com/watch?v=9reQAeUVtgQ',
        26:'https://www.youtube.com/watch?v=BlHLFLNGG8A'
        }
for key in list(video_list.keys()):
    subprocess.call('youtube-dl -f best %s --proxy socks5://127.0.0.1:10808 --default-search "ytsearch" --verbose -o ./MTL_videos/%s.mp4' % (video_list[key],key), shell=True)
