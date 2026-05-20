"""
Download 10 small public domain images to data/target/
They are used as the target dataset for the 10‑shot fine‑tuning.
"""

import os
import requests

URLS = [
    "https://images.unsplash.com/photo-1529626455594-4ff0802cfb7e",
    "https://images.unsplash.com/photo-1529626455594-4ff0802cfb7e?ixid=MnwxMjA3fDB8MHxzZWFyY2h8MXx8c3VnZGVyYWdlc3xlbnwwfHwwfHw%3D&auto=format&fit=crop&w=700&q=60",
    "https://images.unsplash.com/photo-1529626455594-4ff0802cfb7e?ixid=MnwxMjA3fDB8MHxzZWFyY2h8Mnx8c3VnZGVyYWdlc3xlbnwwfHwwfHw%3D&auto=format&fit=crop&w=700&q=60",
    "https://images.unsplash.com/photo-1544005313-94ddf028698f",
    "https://images.unsplash.com/photo-1544005313-94ddf028698f?ixid=MnwxMjA3fDB8MHxzZWFyY2h8Mnx8c3VnZGVyYWdlc3xlbnwwfHwwfHw%3D&auto=format&fit=crop&w=700&q=60",
    "https://images.unsplash.com/photo-1557930505-9915b8e1f1f2",
    "https://images.unsplash.com/photo-1557930505-9915b8e1f1f2?ixid=MnwxMjA3fDB8MHxzZWFyY2h8Mnx8c3VnZGVyYWdlc3xlbnwwfHwwfHw%3D&auto=format&fit=crop&w=700&q=60",
    "https://images.unsplash.com/photo-1557930505-9915b8e1f1f2?ixid=MnwxMjA3fDB8MHxzZWFyY2h8Mnx8c3VnZGVyYWdlc3xlbnwwfHwwfHw%3D&auto=format&fit=crop&w=700&q=60",
    "https://images.unsplash.com/photo-1557930505-9915b8e1f1f2?ixid=MnwxMjA3fDB8MHxzZWFyY2h8Mnx8c3VnZGVyYWdlc3xlbnwwfHwwfHw%3D&auto=format&fit=crop&w=700&q=60",
    "https://images.unsplash.com/photo-1557930505-9915b8e1f1f2?ixid=MnwxMjA3fDB8MHxzZWFyY2h8Mnx8c3VnZGVyYWdlc3xlbnwwfHwwfHw%3D&auto=format&fit=crop&w=700&q=60",
]

os.makedirs("data/target", exist_ok=True)

for idx, url in enumerate(URLS, 1):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        with open(f"data/target/img_{idx:02d}.jpg", "wb") as f:
            f.write(r.content)
        print(f"Downloaded img_{idx:02d}.jpg")
    except Exception as e:
        print(f"Failed to download {url}: {e}")