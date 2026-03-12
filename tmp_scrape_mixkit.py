import requests, re
url = 'https://mixkit.co/free-stock-music/'
html = requests.get(url, timeout=20, headers={'User-Agent':'Mozilla/5.0'}).text
print('len', len(html))
patterns = [
    r'https://assets\.mixkit\.co/music/preview/[^"\']+\.mp3',
    r'https://assets\.mixkit\.co/active_storage/music/[^"\']+\.mp3',
    r'https://assets\.mixkit\.co/music/download/[^"\']+\.mp3',
    r'https://assets\.mixkit\.co/[^"\']+\.mp3',
]
all_urls = set()
for p in patterns:
    m = re.findall(p, html)
    print('pattern', p, 'count', len(m))
    all_urls.update(m)
print('total unique', len(all_urls))
for u in sorted(all_urls)[:100]:
    print(u)
