#pip install ffmpeg moviepy

import moviepy.editor as mp
import re
import os

def mp4tomp3(mp4file,mp3file):
    videoclip=mp.VideoFileClip(mp4file)
    audioclip=videoclip.audio
    audioclip.write_audiofile(mp3file)
    audioclip.close()
    videoclip.close()
    
if __name__ == '__main__':
    
    seq_names = open("aistppfilenames.txt", "r").readlines()
    
    for seq_name in seq_names:
        path_video = os.path.join("Download_aist",seq_name)
        audio_name = seq_name.split('.')[0]+".mp3"
        audio_name = re.sub('c[0-9][0-9]', 'cAll', audio_name)
        path_audio = os.path.join("Download_aistmusic",audio_name)
        mp4tomp3(path_video,path_audio)
        break