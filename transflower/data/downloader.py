import argparse
import os
import moviepy.editor as mp
import re
import sys
import urllib.request

SOURCE_URL = 'https://aistdancedb.ongaaccel.jp/v1.0.0/video/10M/'

def download(video_url, download_folder):
    save_path = os.path.join(download_folder, os.path.basename(video_url))
    urllib.request.urlretrieve(video_url, save_path)
    
def mp4tomp3(mp4file,mp3file):
    videoclip=mp.VideoFileClip(mp4file)
    audioclip=videoclip.audio
    audioclip.write_audiofile(mp3file)
    audioclip.close()
    videoclip.close()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
      description='Scripts for downloading AIST++ videos.')
    parser.add_argument(
      '--download_folder',
      type=str,
      required=True,
      help='where to store AIST++ videos.')
    args = parser.parse_args()
    os.makedirs(args.download_folder, exist_ok=True)

    seq_names = open("aistppfilenames.txt", "r").readlines() #urllib.request.urlopen(LIST_URL)
    i = 0
    for seq_name in seq_names:
        video_urls = os.path.join(SOURCE_URL,seq_name)
        path_video = os.path.join("Download_aist",seq_name)
        audio_name = seq_name.split('.')[0]+".mp3"
        audio_name = re.sub('c[0-9][0-9]', 'cAll', audio_name)
        path_audio = os.path.join("Download_aistmusic",audio_name)
        download(video_urls,download_folder=args.download_folder)
        mp4tomp3(path_video,path_audio)
        sys.stderr.write('\rdownloading %d / %d' % (i + 1, len(seq_names)))
        sys.stderr.write('\ndone.\n')
        i += 1