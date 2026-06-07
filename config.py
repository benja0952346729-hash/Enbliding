import os
from dotenv import load_dotenv

load_dotenv(override=True)

# ══════════════════════════════════════════════════════════════════
# AI API KEYS — እስከ 50 (rotation ይሰራል)
# ══════════════════════════════════════════════════════════════════

API_KEYS = []
for i in range(1, 51):
    key = os.getenv(f"AI_API_KEY_{i}")
    if key and key.strip():
        API_KEYS.append(key.strip())

# ══════════════════════════════════════════════════════════════════
# AI MODEL & BASE URL — .env ላይ ብቻ ይቀየራል
# ══════════════════════════════════════════════════════════════════

BASE_URL = os.getenv("AI_BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL    = os.getenv("AI_MODEL",    "deepseek-ai/deepseek-r1")

# ══════════════════════════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════════════════════════

DATABASE_URL = os.getenv("DATABASE_URL")

# ══════════════════════════════════════════════════════════════════
# GAME CONFIG — ሁሌ fresh ያነባዋል (.env ሲቀየር auto ይሰራል)
# ══════════════════════════════════════════════════════════════════

def get_game_config() -> dict:
    load_dotenv(override=True)
    return {
        # Board
        "slots_total":         int(os.getenv("SLOTS_TOTAL",         100)),
        "slots_per_person":    int(os.getenv("SLOTS_PER_PERSON",    5)),

        # Prices
        "price_full":          int(os.getenv("PRICE_FULL",          400)),
        "price_half":          int(os.getenv("PRICE_HALF",          200)),

        # Prizes
        "prize_1st":           int(os.getenv("PRIZE_1ST",           5000)),
        "prize_2nd":           int(os.getenv("PRIZE_2ND",           1000)),
        "prize_3rd":           int(os.getenv("PRIZE_3RD",           400)),
        "winners_count":       int(os.getenv("WINNERS_COUNT",       3)),

        # Timing
        "warning_minutes":     int(os.getenv("WARNING_MINUTES",     2)),

        # UI
        "low_slots_threshold": int(os.getenv("LOW_SLOTS_THRESHOLD", 7)),

        # Payment Accounts
        "cbe_account":         os.getenv("CBE_ACCOUNT"),
        "cbe_name":            os.getenv("CBE_NAME"),
        "awash_account":       os.getenv("AWASH_ACCOUNT"),
        "dashen_account":      os.getenv("DASHEN_ACCOUNT"),
        "tele_birr":           os.getenv("TELE_BIRR"),
    }

# ══════════════════════════════════════════════════════════════════
# API KEY ROTATION
# ══════════════════════════════════════════════════════════════════

_current_key_index = 0

def get_api_key() -> str:
    if not API_KEYS:
        raise Exception("❌ API key የለም! .env ላይ AI_API_KEY_1 ጨምር")
    return API_KEYS[_current_key_index]

def rotate_key():
    global _current_key_index
    _current_key_index = (_current_key_index + 1) % len(API_KEYS)
    print(f"🔄 Key {_current_key_index + 1}/{len(API_KEYS)} ላይ ተዛወረ")

def get_key_status() -> str:
    if not API_KEYS:
        return "❌ API key የለም"
    return f"✅ {len(API_KEYS)} keys — አሁን: {_current_key_index + 1}"
