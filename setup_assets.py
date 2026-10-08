import os
import urllib.request

# 1. Create the emojis directory if it doesn't exist
os.makedirs('emojis', exist_ok=True)

# 2. Define high-quality emoji URLs (Fluent Emoji set)
emoji_urls = {
    "Angry": "https://fonts.gstatic.com/s/e/notoemoji/latest/1f621/512.png",
    "Disgusted": "https://fonts.gstatic.com/s/e/notoemoji/latest/1f922/512.png",
    "Fearful": "https://fonts.gstatic.com/s/e/notoemoji/latest/1f628/512.png",
    "Happy": "https://fonts.gstatic.com/s/e/notoemoji/latest/1f60a/512.png",
    "Neutral": "https://fonts.gstatic.com/s/e/notoemoji/latest/1f610/512.png",
    "Sad": "https://fonts.gstatic.com/s/e/notoemoji/latest/1f622/512.png",
    "Surprised": "https://fonts.gstatic.com/s/e/notoemoji/latest/1f632/512.png"
}

print("Starting asset download...")
for emotion, url in emoji_urls.items():
    filename = f"emojis/{emotion}.png"
    try:
        urllib.request.urlretrieve(url, filename)
        print(f" Successfully downloaded: {filename}")
    except Exception as e:
        print(f" Failed to download {emotion}: {e}")

print("\nAll set! You now have the 'emojis/' folder ready.")