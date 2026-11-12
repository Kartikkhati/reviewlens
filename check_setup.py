# check_setup.py - confirms packages are installed and the LLM key works
import os
import sys

from dotenv import load_dotenv

load_dotenv()  # reads LLM_* settings from the .env file into environment variables

print("Python:", sys.version.split()[0])

# 1. Are all libraries installed?
import pandas, openai, streamlit, matplotlib, google_play_scraper  # noqa: F401
print("All packages imported OK")

# 2. Are the environment variables set? (we never print the key itself)
missing = [n for n in ["LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"] if not os.getenv(n)]
for name in ["LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"]:
    print(f"{name}: {'MISSING' if name in missing else 'set'}")
if missing:
    sys.exit("Set the missing environment variables, then run this again.")

# 3. Can we actually talk to the LLM?
from openai import OpenAI
client = OpenAI(api_key=os.getenv("LLM_API_KEY"), base_url=os.getenv("LLM_BASE_URL"))
reply = client.chat.completions.create(
    model=os.getenv("LLM_MODEL"),
    messages=[{"role": "user", "content": "Reply with exactly: setup works"}],
    extra_body={"reasoning_effort": "low"},  # gpt-oss is a reasoning model; keep thinking short
)
print("LLM says:", reply.choices[0].message.content)
