"""
Instagram Reels Archiver
Pipeline: yt-dlp (download) -> Whisper (transcription) -> Groq (AI summarization & tagging)

Given a JSON export of your saved Instagram posts, this script downloads the audio
of each reel, transcribes it locally with Whisper, and uses the Groq API to produce
a structured summary (title, key points, category, tags). Results are saved
incrementally to output.json and rendered into a readable report.md.
"""

import json
import os
import re
import time
import random
import yt_dlp
import whisper
from groq import Groq
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# --- Groq configuration ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Make sure your .env file exists and is configured correctly.")

groq_client = Groq(api_key=GROQ_API_KEY)
# llama-3.3-70b-versatile = higher quality, 1,000 req/day free tier
# llama-3.1-8b-instant   = lighter/faster, 14,400 req/day free tier (better if you process many reels/day)
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# --- Whisper configuration ---
WHISPER_MODEL_SIZE = "medium"

# --- File paths ---
JSON_EXPORT_PATH = "saved_posts.json"
OUTPUT_JSON_PATH = "output.json"
REPORT_MD_PATH = "report.md"

# --- Retry / rate-limit handling ---
MAX_RETRIES = 5
BASE_BACKOFF_SECONDS = 2


def load_existing_results(json_path: str = OUTPUT_JSON_PATH) -> List[Dict]:
    """Load previous results so the script can resume where it left off."""
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"{json_path} exists but is corrupted. It will be overwritten.")
    return []


def extract_urls_from_json(json_path: str) -> List[str]:
    """Extract unique Instagram reel/post URLs from the exported JSON file."""
    urls: List[str] = []
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            content = f.read()
            # Regex match for Instagram reel/post links inside the exported JSON
            raw_urls = re.findall(r"https?://(?:www\.)?instagram\.com/(?:reel|p)/[\w-]+", content)
            # Deduplicate while preserving original order (list(set()) would lose order)
            urls = list(dict.fromkeys(raw_urls))
    except FileNotFoundError:
        print(f"File {json_path} not found.")
    return urls


def download_instagram_audio(url: str, output_name: str) -> Dict[str, Optional[str]]:
    """Download and extract the audio track of an Instagram reel via yt-dlp."""
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
        "outtmpl": f"{output_name}.%(ext)s",
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
        "noplaylist": True,  # avoid unexpected playlist behaviour
    }

    result: Dict[str, Optional[str]] = {"audio_path": None, "caption": "", "error": None}

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            expected_path = f"{output_name}.mp3"

            if os.path.exists(expected_path):
                result["audio_path"] = expected_path
            else:
                result["error"] = "No audio extracted (the video might be silent)."

            result["caption"] = info.get("description", "")
    except Exception as e:
        result["error"] = f"yt-dlp error: {str(e)}"

    return result


def transcribe_audio(audio_path: Optional[str], model) -> Dict:
    """Transcribe an audio file with Whisper, handling missing files and runtime errors."""
    if not audio_path or not os.path.exists(audio_path):
        return {"text": "", "language": None, "error": "Audio file not available."}

    print("Transcribing audio...")
    try:
        result = model.transcribe(audio_path)
        return {
            "text": result.get("text", "").strip(),
            "language": result.get("language", "unknown"),
            "error": None,
        }
    except Exception as e:
        return {"text": "", "language": None, "error": f"Whisper exception: {str(e)}"}


def build_prompt(transcription: str, caption: str) -> str:
    """Build the analysis prompt sent to the LLM."""
    return f"""
    You are an expert assistant. Analyze the following video transcription and its caption.
    The source video can be in any language, but your output MUST BE EXCLUSIVELY IN ITALIAN.

    Raw transcription: "{transcription}"
    Original caption: "{caption}"

    Tasks:
    1. Filter out errors and hallucinations caused by phonetics or background music.
    2. Write an essential summary (2-3 sentences) of the key concept.
    3. Extract practical key points (ingredients/doses, steps, commands, place names).
    4. Assign a macro-category and 3-5 descriptive tags.

    Return the output strictly as JSON with the following structure:
    {{
        "titolo_generato": "Short title",
        "riassunto": "2-3 sentence summary",
        "punti_chiave": ["Point 1", "Point 2"],
        "macro_categoria": "Category",
        "tags": ["tag1", "tag2"]
    }}
    """


def process_with_groq(transcription: str, caption: str) -> Dict:
    """Use Groq to clean up, summarize and extract structured data as JSON.

    Retries with exponential backoff + jitter on transient/rate-limit errors,
    so a single 429 doesn't cost you the whole item.
    """
    prompt = build_prompt(transcription, caption)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            error_str = str(e)
            is_rate_limit = "rate_limit" in error_str.lower() or "429" in error_str

            if attempt == MAX_RETRIES:
                print(f"Groq API error (final attempt {attempt}/{MAX_RETRIES}): {e}")
                return {"error": error_str}

            wait_time = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 1)
            if is_rate_limit:
                wait_time *= 2  # back off harder on explicit rate-limit errors
            print(f"Groq API error (attempt {attempt}/{MAX_RETRIES}): {e}. Retrying in {wait_time:.1f}s...")
            time.sleep(wait_time)

    return {"error": "Max retries exceeded."}


def save_outputs(results: List[Dict]):
    """Save results to JSON and render the formatted Markdown report."""
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    with open(REPORT_MD_PATH, "w", encoding="utf-8") as f:
        f.write("# Saved Reels Archive\n\n")

        # Collect failed items to list at the end of the document
        failed_items = []

        for item in results:
            if item.get("error"):
                failed_items.append(item)
                continue

            ai_data = item.get("ai_data", {})
            whisper_info = item.get("whisper_data", {})
            source_language = whisper_info.get("language", "N/A")

            f.write(f"## {ai_data.get('titolo_generato', 'Untitled')}\n")
            f.write(f"**URL:** {item['url']} | **Detected Language:** {source_language}\n")
            f.write(
                f"**Category:** {ai_data.get('macro_categoria', 'N/A')} | "
                f"**Tags:** {', '.join(ai_data.get('tags', []))}\n\n"
            )

            f.write(f"### Summary\n{ai_data.get('riassunto', '')}\n\n")

            key_points = ai_data.get("punti_chiave", [])
            if key_points:
                f.write("### Practical Points\n")
                for point in key_points:
                    f.write(f"- {point}\n")
                f.write("\n")

            corrected_transcript = ai_data.get("trascrizione_corretta", "")
            if corrected_transcript:
                f.write(
                    f"<details><summary><b>Show Full Transcript</b></summary>\n\n"
                    f"{corrected_transcript}\n</details>\n\n"
                )

            f.write("---\n\n")

        # Errors section
        if failed_items:
            f.write("# ⚠️ Unprocessed Videos (Errors)\n\n")
            for err in failed_items:
                f.write(f"- **URL:** {err['url']}\n  - *Cause:* {err.get('error', 'Unknown error')}\n")


def main():
    urls = extract_urls_from_json(JSON_EXPORT_PATH)

    if not urls:
        print("No URLs found. Check the structure of your JSON export.")
        return

    existing_results = load_existing_results()
    processed_urls = {item["url"] for item in existing_results if "url" in item}
    urls_to_process = [url for url in urls if url not in processed_urls]

    print(f"Total URLs: {len(urls)} | Already processed: {len(processed_urls)} | To do: {len(urls_to_process)}")

    if not urls_to_process:
        print("All videos have already been processed.")
        return

    print(f"\nLoading Whisper model '{WHISPER_MODEL_SIZE}'...")
    whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)

    final_results = existing_results.copy()

    for i, url in enumerate(urls_to_process):
        print(f"\n[{i + 1}/{len(urls_to_process)}] Processing: {url}")
        temp_audio_name = f"temp_audio_{i}"

        # 1. Download
        dl_result = download_instagram_audio(url, temp_audio_name)

        if dl_result["error"] and not dl_result["caption"]:
            print(f"Skipped: {dl_result['error']}")
            final_results.append({"url": url, "error": dl_result["error"]})
        else:
            # 2. Transcription
            whisper_result = transcribe_audio(dl_result["audio_path"], whisper_model)

            if whisper_result["error"] and not dl_result["caption"].strip():
                print(f"No content: {whisper_result['error']}")
                final_results.append({"url": url, "error": whisper_result["error"]})
            else:
                # 3. Groq processing
                print("Running AI analysis...")
                ai_data = process_with_groq(whisper_result["text"], dl_result["caption"])

                if "error" in ai_data:
                    final_results.append({"url": url, "error": f"LLM error: {ai_data['error']}"})
                else:
                    final_results.append(
                        {
                            "url": url,
                            "whisper_data": whisper_result,
                            "ai_data": ai_data,
                            "raw_caption": dl_result["caption"],
                        }
                    )

        # Clean up local audio file
        if dl_result["audio_path"] and os.path.exists(dl_result["audio_path"]):
            os.remove(dl_result["audio_path"])

        # 4. Checkpoint (incremental save after every iteration)
        save_outputs(final_results)

    print(f"\nDone. Data saved to {OUTPUT_JSON_PATH} and {REPORT_MD_PATH}.")


if __name__ == "__main__":
    main()