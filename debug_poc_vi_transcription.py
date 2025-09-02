import os
import asyncio
from typing import Dict, List
from datetime import datetime
import httpx
from dotenv import load_dotenv


def hhmmss_to_seconds(ts: str) -> int:
    try:
        if ":" in ts:
            parts = ts.split(":")
            if len(parts) == 3:
                h, m, s = map(float, parts)
                return int(h * 3600 + m * 60 + s)
            if len(parts) == 2:
                m, s = map(float, parts)
                return int(m * 60 + s)
        return int(float(ts))
    except Exception:
        return 0


def seconds_to_hhmmss(sec: int) -> str:
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def extract_transcript_with_timestamps(index_json: Dict) -> List[Dict]:
    """
    Pulls transcript items and maps to: text, start_time, end_time, start_seconds, end_seconds, duration, confidence.
    """
    items = (
        index_json
        .get("videos", [{}])[0]
        .get("insights", {})
        .get("transcript", [])
    )
    out: List[Dict] = []
    for it in items:
        text = it.get("text", "")
        inst = it.get("instances", [])
        if not text or not inst:
            continue
        first = inst[0]
        start = first.get("start", "00:00:00")
        end = first.get("end", "00:00:00")
        s_sec = hhmmss_to_seconds(start)
        e_sec = hhmmss_to_seconds(end)
        out.append({
            "text": text,
            "start_time": start,
            "end_time": end,
            "start_seconds": s_sec,
            "end_seconds": e_sec,
            "duration": max(0, e_sec - s_sec),
            "confidence": it.get("confidence", 0.9)
        })
    return out


async def fetch_index_by_video_id(account_id: str, location: str, video_id: str, access_token: str) -> Dict:
    """
    Minimal GET to: /{location}/Accounts/{account_id}/Videos/{video_id}/Index?accessToken=...
    """
    url = f"https://api.videoindexer.ai/{location}/Accounts/{account_id}/Videos/{video_id}/Index"
    params = {"accessToken": access_token}
    # For production, keep verify=True (default). If you have local trust issues, temporarily set verify=False.
    async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


async def main():
    load_dotenv()

    account_id = os.getenv("VIDEO_INDEXER_ACCOUNT_ID", "").strip()
    location = os.getenv("VIDEO_INDEXER_LOCATION", "").strip()

    # אלה הפרמטרים שתצטרכי לספק
    video_id = input("Enter video_id: ").strip()
    access_token = input("Enter access_token: ").strip()

    if not (account_id and location and video_id and access_token):
        raise RuntimeError(
            "Missing required parameters. Please provide video_id and access_token, and ensure VIDEO_INDEXER_ACCOUNT_ID and VIDEO_INDEXER_LOCATION are set in .env")

    print(f"[{datetime.now().isoformat()}] Fetching index for video_id={video_id} ...")
    index_json = await fetch_index_by_video_id(account_id, location, video_id, access_token)

    state = index_json.get("state", "Unknown")
    print(f"Video state: {state}")
    if state != "Processed":
        print("Warning: Video is not Processed yet. You may see a partial or empty transcript.")

    segments = extract_transcript_with_timestamps(index_json)
    print(f"Found {len(segments)} transcript segments.")

    # Write a simple flat transcript with timestamps
    out_txt = "transcript.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(f"[{seg['start_time']}] {seg['text']}\n")
    print(f"Wrote transcript to {out_txt}")

    # Also save the full JSON for debugging
    import json
    with open("video_index.json", "w", encoding="utf-8") as f:
        json.dump(index_json, f, ensure_ascii=False, indent=2)
    print("Saved full index JSON to video_index.json")


if __name__ == "__main__":
    asyncio.run(main())
