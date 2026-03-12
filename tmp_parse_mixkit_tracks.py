import requests, re
from bs4 import BeautifulSoup

html = requests.get('https://mixkit.co/free-stock-music/', headers={'User-Agent':'Mozilla/5.0'}, timeout=20).text
soup = BeautifulSoup(html, 'html.parser')

cards = soup.select('div.item-grid-card')
tracks = []
for c in cards:
    div = c.select_one('[data-audio-player-preview-url-value]')
    if not div:
        continue
    url = div.get('data-audio-player-preview-url-value','').strip()
    item_id = div.get('data-audio-player-item-id-value','').strip()
    title_el = c.select_one('h2.item-grid-card__title')
    title = title_el.get_text(' ', strip=True) if title_el else ''
    dur_el = c.select_one('[data-test-id="duration"]')
    dur = dur_el.get_text(' ', strip=True) if dur_el else ''
    tags = [a.get_text(' ', strip=True).lower() for a in c.select('.meta-links__link') if a.get_text(' ', strip=True)]
    tracks.append({'id':item_id,'title':title,'dur':dur,'url':url,'tags':tags})

print('tracks', len(tracks))
for t in tracks:
    print(f"{t['id']} | {t['title']} | {t['dur']} | {','.join(t['tags'])} | {t['url']}")
