=======================================
Instagram Reels Archiver
=======================================

A tool that processes a JSON export of your saved Instagram posts,
downloads the audio of each reel, transcribes it locally with Whisper,
and uses the Groq API to generate a structured AI summary (title, key
points, category, tags) for each one.

Pipeline: yt-dlp (download) -> Whisper (transcription) -> Groq (AI
summarization & tagging)

--------------------
FEATURES
--------------------
- Extracts unique Instagram reel/post URLs from a saved-posts JSON export
- Downloads the audio track of each reel via yt-dlp
- Transcribes audio locally with Whisper (audio never leaves your machine
  for transcription)
- Uses the Groq API (LLM) to clean up the transcript and generate a
  summary, practical key points, a category and tags (output in Italian)
- Incremental / resumable: URLs already present in output.json are
  skipped, so you can stop and restart the script safely
- Automatic retry with exponential backoff + jitter on Groq rate-limit
  or transient errors
- Deletes the temporary audio file after each reel is processed
- Produces two outputs: output.json (structured data) and report.md
  (human-readable report)

--------------------
PROJECT STRUCTURE
--------------------
main.py                Entry point, contains the full pipeline
requirements.txt       Python package dependencies
.env                    API keys and configuration (not committed)
saved_posts.json        Your Instagram data export (input, not included)
output.json             Generated results (created automatically)
report.md               Generated Markdown report (created automatically)

--------------------
REQUIREMENTS
--------------------
- Python 3.9+
- FFmpeg installed and available on your PATH (required by yt-dlp to
  extract the audio track as MP3)
- A free Groq API key (https://console.groq.com)
- Python packages (listed in requirements.txt):
    yt-dlp
    openai-whisper
    groq
    python-dotenv

Install all required packages with:
    pip install -r requirements.txt

Note: Whisper needs an ffmpeg-compatible setup and, the first time it
runs, will automatically download the "medium" model (about 1.5 GB).
Transcription works on CPU but is noticeably faster with a GPU.

--------------------
SETUP
--------------------
1. Create a free API key at https://console.groq.com
2. In the project root, create a file named ".env" with:

    GROQ_API_KEY=your_api_key_here
    GROQ_MODEL=llama-3.3-70b-versatile

   GROQ_MODEL is optional (defaults to "llama-3.3-70b-versatile" if not
   set). You can switch to "llama-3.1-8b-instant" for a faster, lighter
   model with a higher daily free-tier request limit.

3. Export your saved Instagram posts using Instagram's "Download Your
   Information" tool and place the resulting JSON file in the project
   root as "saved_posts.json". The script scans this file for links in
   the form instagram.com/reel/... and instagram.com/p/...

--------------------
USAGE
--------------------
Run the script from the project root:

    python main.py

What happens:
1. URLs are extracted from saved_posts.json
2. URLs already present in output.json are skipped (resume support)
3. The Whisper model ("medium") is loaded
4. For each new URL: download audio -> transcribe -> summarize via Groq
5. Results are saved to output.json after every single item (checkpoint)
6. output.json and report.md are (re)generated at the end

--------------------
OUTPUT FILES
--------------------
- output.json: full structured results per URL, including the raw
  Whisper transcription, detected language, AI-generated data, and any
  errors encountered
- report.md: a readable Markdown report with, for each reel, its title,
  category, tags, summary, key points and a collapsible full transcript;
  failed/unprocessed videos are listed separately at the end

--------------------
NOTES / LIMITATIONS
--------------------
- The AI-generated summary is currently fixed to be written in Italian,
  regardless of the reel's original language (this is set in the prompt
  inside main.py and can be changed there).
- The script will not start if GROQ_API_KEY is missing from the .env file.
- Temporary audio files are deleted right after each reel is processed,
  so only the transcript and AI output are kept.
- Make sure your use of this tool complies with Instagram's Terms of
  Service and that you only process content you have the rights to.