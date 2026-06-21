import os


KEYS = [
    "OPENAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "TRACK2_OPENAI_BASE_URL",
    "TRACK2_DEEPSEEK_BASE_URL",
    "OPENAI_BASE_URL",
    "DEEPSEEK_BASE_URL",
]


for key in KEYS:
    print(f"{key}_PRESENT={bool(os.environ.get(key))}")
