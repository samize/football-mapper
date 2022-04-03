# Bryant Cornwell
# January 3rd, 2022
"""
Download Youtube video using given url
"""

# from urllib.parse import urlparse
import requests
from pytube import YouTube

def download_video(video_url):
    # Add video to memory.
    #video_file = requests.get(video_url, stream=True)
    # Convert and export video to mp4
    #with open("video_name.mp4", "wb+") as file:
    #    file.write(video_file)
    video = YouTube(str(video_url))
    video = video.streams.get_by_itag(22)
    # Filter for the mp4 file
    #video.filter(progressive=True, file_extension='mp4')
    # Choose the best resolution
    #video.order_by('resolution')
    #video.desc()
    #video.first()
    # download the video
    video.download()
    # get url for thumbnail
    print("Video Thumbnail URL:", YouTube(str(video_url)).thumbnail_url)


if __name__ == "__main__":
    url = input("Please enter the url to the desired Youtube video: ")
    download_video(url)