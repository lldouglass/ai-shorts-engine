"""Post Range Rover video to YouTube Shorts + Facebook Reels + Instagram Reels."""
import sys
sys.stdout.reconfigure(line_buffering=True)

import json
import os
import time
import urllib.request
import urllib.parse
import subprocess

VIDEO_PATH = r"C:\Users\Logan\Downloads\ai-shorts-engine\output\car_videos\range_rover_heygen_final.mp4"
BRAND = "car"

TITLE = "The Range Rover scores a 2.0 out of 5 for reliability"

DESCRIPTION = (
    "The Range Rover has a 2.0 out of 5.0 reliability rating on RepairPal, "
    "ranking 15th out of 19 luxury full-size SUVs. Average repair cost? $1,258/year. "
    "Consumer Reports flagged 5 active recalls on the 2024 model. "
    "Air suspension issues, electrical gremlins, and oil leaks that can cause engine fires. "
    "Starting MSRP over $100K.\n\n"
    "Data from RepairPal, Consumer Reports, and NHTSA.\n\n"
    "#RangeRover #CarReliability #LandRover #LuxurySUV #UsedCars #CarData #NHTSA #Recalls "
    "#CarLifespanCheck #UnreliableCars #CostOfOwnership #CarBuying"
)

SHORT_CAPTION = (
    "Range Rover: 2.0/5 reliability. $1,258/year in repairs. 5 active recalls. "
    "15th out of 19 luxury SUVs. At least it looks good in the shop.\n\n"
    "Data: RepairPal, Consumer Reports, NHTSA\n\n"
    "#RangeRover #CarReliability #LandRover #LuxurySUV #UsedCars "
    "#CarData #NHTSA #Recalls #CarLifespanCheck #UnreliableCars"
)


def post_youtube(video_path, title, description, brand="car"):
    print("\n=== YouTube Shorts ===")
    tokens = json.load(open("config/yt_channel_tokens.json"))
    token_info = tokens[brand]
    refresh_token = token_info["refresh_token"]

    CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
    CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")


    data = urllib.parse.urlencode({
        "client_id": CLIENT_ID, "client_secret": CLIENT_SECRET,
        "refresh_token": refresh_token, "grant_type": "refresh_token",
    }).encode()
    req = urllib.request.Request("https://oauth2.googleapis.com/token", data=data, method="POST")
    with urllib.request.urlopen(req) as resp:
        access_token = json.loads(resp.read())["access_token"]

    print(f"  Channel: {token_info['channel_title']} ({token_info['channel_id']})")

    metadata = json.dumps({
        "snippet": {
            "title": title + " #Shorts",
            "description": description,
            "tags": ["Range Rover", "reliability", "car data", "luxury SUV", "recalls", "NHTSA",
                     "car lifespan check", "used cars", "cost of ownership", "Land Rover"],
            "categoryId": "2",
        },
        "status": {"privacyStatus": "public", "selfDeclaredMadeForKids": False},
    }).encode()

    file_size = os.path.getsize(video_path)
    init_url = "https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status"
    init_req = urllib.request.Request(init_url, data=metadata, method="POST", headers={
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json; charset=UTF-8",
        "X-Upload-Content-Length": str(file_size),
        "X-Upload-Content-Type": "video/mp4",
    })
    with urllib.request.urlopen(init_req) as resp:
        upload_url = resp.headers.get("Location")

    with open(video_path, "rb") as f:
        video_data = f.read()
    upload_req = urllib.request.Request(upload_url, data=video_data, method="PUT", headers={
        "Content-Type": "video/mp4", "Content-Length": str(file_size),
    })
    with urllib.request.urlopen(upload_req) as resp:
        result = json.loads(resp.read())

    video_id = result.get("id", "?")
    print(f"  Uploaded! Video ID: {video_id}")
    print(f"  URL: https://youtube.com/shorts/{video_id}")
    return video_id


def post_facebook(video_path, description, brand="car"):
    print("\n=== Facebook Reels ===")
    fb_tokens = json.load(open("config/fb_tokens.json"))
    page_id = "931510130055658" if brand == "car" else "1033367676524648"
    page_token = fb_tokens["page_tokens"][page_id]["access_token"]
    page_name = fb_tokens["page_tokens"][page_id]["name"]
    print(f"  Page: {page_name} ({page_id})")

    file_size = os.path.getsize(video_path)
    init_url = f"https://graph.facebook.com/v25.0/{page_id}/video_reels"
    init_data = urllib.parse.urlencode({
        "upload_phase": "start", "access_token": page_token,
    }).encode()
    req = urllib.request.Request(init_url, data=init_data, method="POST")
    with urllib.request.urlopen(req) as resp:
        init_result = json.loads(resp.read())

    video_id = init_result.get("video_id")
    upload_url = init_result.get("upload_url")
    print(f"  Video ID: {video_id}")
    if not upload_url:
        print(f"  No upload URL. Response: {init_result}")
        return None

    with open(video_path, "rb") as f:
        video_data = f.read()
    upload_req = urllib.request.Request(upload_url, data=video_data, method="POST", headers={
        "Authorization": f"OAuth {page_token}",
        "offset": "0", "file_size": str(file_size),
        "Content-Type": "application/octet-stream",
    })
    with urllib.request.urlopen(upload_req) as resp:
        upload_result = json.loads(resp.read())
    print(f"  Upload: {upload_result}")

    publish_url = f"https://graph.facebook.com/v25.0/{page_id}/video_reels"
    publish_data = urllib.parse.urlencode({
        "upload_phase": "finish", "video_id": video_id,
        "title": TITLE, "description": description,
        "access_token": page_token,
    }).encode()
    pub_req = urllib.request.Request(publish_url, data=publish_data, method="POST")
    with urllib.request.urlopen(pub_req) as resp:
        pub_result = json.loads(resp.read())
    print(f"  Published! Result: {pub_result}")
    return video_id


def post_instagram(video_path, caption, brand="car"):
    print("\n=== Instagram Reels ===")
    fb_tokens = json.load(open("config/fb_tokens.json"))
    page_id = "931510130055658" if brand == "car" else "1033367676524648"
    page_info = fb_tokens["page_tokens"][page_id]
    page_token = page_info["access_token"]
    ig_id = page_info.get("instagram_business_id")
    if not ig_id:
        print("  ERROR: No Instagram Business ID!")
        return None
    print(f"  IG Business ID: {ig_id}")

    print("  Uploading video to temp host...")
    result = subprocess.run(
        ["curl", "-s", "-F", f"file=@{video_path}", "https://0x0.st"],
        capture_output=True, text=True, timeout=60,
    )
    public_url = result.stdout.strip()
    if not public_url.startswith("http"):
        print(f"  Upload failed: {result.stdout} {result.stderr}")
        return None
    print(f"  Public URL: {public_url}")

    create_url = f"https://graph.facebook.com/v25.0/{ig_id}/media"
    create_data = urllib.parse.urlencode({
        "media_type": "REELS", "video_url": public_url,
        "caption": caption, "access_token": page_token,
    }).encode()
    req = urllib.request.Request(create_url, data=create_data, method="POST")
    with urllib.request.urlopen(req) as resp:
        container = json.loads(resp.read())
    container_id = container.get("id")
    print(f"  Container ID: {container_id}")

    print("  Waiting for IG processing...")
    for i in range(30):
        time.sleep(5)
        status_url = f"https://graph.facebook.com/v25.0/{container_id}?fields=status_code,status&access_token={page_token}"
        with urllib.request.urlopen(status_url) as resp:
            status = json.loads(resp.read())
        code = status.get("status_code", "UNKNOWN")
        print(f"    Status: {code}")
        if code == "FINISHED":
            break
        elif code == "ERROR":
            print(f"  ERROR: {status}")
            return None

    publish_url = f"https://graph.facebook.com/v25.0/{ig_id}/media_publish"
    publish_data = urllib.parse.urlencode({
        "creation_id": container_id, "access_token": page_token,
    }).encode()
    pub_req = urllib.request.Request(publish_url, data=publish_data, method="POST")
    with urllib.request.urlopen(pub_req) as resp:
        pub_result = json.loads(resp.read())
    ig_media_id = pub_result.get("id")
    print(f"  Published! IG Media ID: {ig_media_id}")
    return ig_media_id


if __name__ == "__main__":
    print(f"Video: {VIDEO_PATH}")
    print(f"Size: {os.path.getsize(VIDEO_PATH) / 1024 / 1024:.1f}MB")
    print(f"Brand: {BRAND}")
    results = {}

    for name, fn, args in [
        ("youtube", post_youtube, (VIDEO_PATH, TITLE, DESCRIPTION, BRAND)),
        ("facebook", post_facebook, (VIDEO_PATH, SHORT_CAPTION, BRAND)),
        ("instagram", post_instagram, (VIDEO_PATH, SHORT_CAPTION, BRAND)),
    ]:
        try:
            r = fn(*args)
            results[name] = {"id": r}
            if name == "youtube":
                results[name]["url"] = f"https://youtube.com/shorts/{r}"
        except Exception as e:
            print(f"  {name} FAILED: {e}")
            results[name] = {"error": str(e)}

    print("\n=== RESULTS ===")
    print(json.dumps(results, indent=2))
