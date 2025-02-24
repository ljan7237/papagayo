import os
url = "YOUR URL"
os.system(f"yt-dlp -x --audio-format mp3 --audio-quality 0 {url}")
