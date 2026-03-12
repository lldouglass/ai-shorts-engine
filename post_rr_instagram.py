"""Post Range Rover video to Instagram Reels via Graph API."""
import json
import time
import urllib.request
import urllib.parse

API_VERSION = "v25.0"
BASE = f"https://graph.facebook.com/{API_VERSION}"

VIDEO_URL = "https://www.carlifespancheck.com/videos/range_rover_heygen.mp4"
CAPTION = (
    "Range Rover: 2.0/5 reliability. $1,258/year in repairs. 5 active recalls. "
    "15th out of 19 luxury SUVs. At least it looks good in the shop.\n\n"
    "Data: RepairPal, Consumer Reports, NHTSA\n\n"
    "#RangeRover #CarReliability #LandRover #LuxurySUV #UsedCars "
    "#CarData #NHTSA #Recalls #CarLifespanCheck #UnreliableCars"
)

fb_tokens = json.load(open("config/fb_tokens.json"))
page_id = "931510130055658"
page_token = fb_tokens["page_tokens"][page_id]["access_token"]
ig_id = fb_tokens["page_tokens"][page_id].get("instagram_business_id")
print(f"IG Business ID: {ig_id}")

# Create container
print("Creating media container...")
create_data = urllib.parse.urlencode({
    "media_type": "REELS",
    "video_url": VIDEO_URL,
    "caption": CAPTION,
    "access_token": page_token,
}).encode()
req = urllib.request.Request(f"{BASE}/{ig_id}/media", data=create_data, method="POST")
with urllib.request.urlopen(req) as resp:
    container = json.loads(resp.read())
container_id = container.get("id")
print(f"Container ID: {container_id}")

# Wait for processing
print("Waiting for processing...")
for i in range(40):
    time.sleep(5)
    status_url = f"{BASE}/{container_id}?fields=status_code,status&access_token={page_token}"
    with urllib.request.urlopen(status_url) as resp:
        status = json.loads(resp.read())
    code = status.get("status_code", "UNKNOWN")
    print(f"  [{(i+1)*5}s] Status: {code}")
    if code == "FINISHED":
        break
    elif code == "ERROR":
        print(f"  ERROR: {status}")
        raise SystemExit(1)

# Publish
print("Publishing...")
pub_data = urllib.parse.urlencode({
    "creation_id": container_id,
    "access_token": page_token,
}).encode()
pub_req = urllib.request.Request(f"{BASE}/{ig_id}/media_publish", data=pub_data, method="POST")
with urllib.request.urlopen(pub_req) as resp:
    result = json.loads(resp.read())
print(f"Published! IG Media ID: {result.get('id')}")
