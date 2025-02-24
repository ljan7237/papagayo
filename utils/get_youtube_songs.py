import os
# bohemian rhapsody: https://www.youtube.com/watch?v=fJ9rUzIMcZQ
url = "https://www.youtube.com/watch?v=fJ9rUzIMcZQ"#"https://www.youtube.com/watch?v=mrudT410TAI"#flower"https://www.youtube.com/watch?v=6A2V9Bu80J4"#plumfairy"https://www.youtube.com/watch?v=o0q2yh_VDpU"#at last"https://www.youtube.com/watch?v=1qJU8G7gR_g" # etta james
os.system(f"yt-dlp -x --audio-format mp3 --audio-quality 0 {url}")
