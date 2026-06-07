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
# GAME CONFIG — DB ያነባል → .env fallback → defaults
# ══════════════════════════════════════════════════════════════════

_DEFAULTS = {
    "slots_total":         100,
    "slots_per_person":    5,
    "price_full":          400,
    "price_half":          200,
    "prize_1st":           5000,
    "prize_2nd":           1000,
    "prize_3rd":           400,
    "winners_count":       3,
    "warning_minutes":     2,
    "low_slots_threshold": 7,
    "cbe_account":         None,
    "cbe_name":            None,
    "awash_account":       None,
    "dashen_account":      None,
    "tele_birr":           None,
}

def _env_fallback() -> dict:
    """DB ካልሆነ .env ያነባል"""
    load_dotenv(override=True)
    return {
        "slots_total":         int(os.getenv("SLOTS_TOTAL",         _DEFAULTS["slots_total"])),
        "slots_per_person":    int(os.getenv("SLOTS_PER_PERSON",    _DEFAULTS["slots_per_person"])),
        "price_full":          int(os.getenv("PRICE_FULL",          _DEFAULTS["price_full"])),
        "price_half":          int(os.getenv("PRICE_HALF",          _DEFAULTS["price_half"])),
        "prize_1st":           int(os.getenv("PRIZE_1ST",           _DEFAULTS["prize_1st"])),
        "prize_2nd":           int(os.getenv("PRIZE_2ND",           _DEFAULTS["prize_2nd"])),
        "prize_3rd":           int(os.getenv("PRIZE_3RD",           _DEFAULTS["prize_3rd"])),
        "winners_count":       int(os.getenv("WINNERS_COUNT",       _DEFAULTS["winners_count"])),
        "warning_minutes":     int(os.getenv("WARNING_MINUTES",     _DEFAULTS["warning_minutes"])),
        "low_slots_threshold": int(os.getenv("LOW_SLOTS_THRESHOLD", _DEFAULTS["low_slots_threshold"])),
        "cbe_account":         os.getenv("CBE_ACCOUNT"),
        "cbe_name":            os.getenv("CBE_NAME"),
        "awash_account":       os.getenv("AWASH_ACCOUNT"),
        "dashen_account":      os.getenv("DASHEN_ACCOUNT"),
        "tele_birr":           os.getenv("TELE_BIRR"),
    }

def get_game_config() -> dict:
    """
    1. DB game_config table ያነባል (Repo 1 ያስቀምጣል)
    2. DB ካልሆነ → .env fallback
    3. .env ካልሆነ → defaults
    """
    if not DATABASE_URL:
        return _env_fallback()

    try:
        import psycopg2
        import psycopg2.extras
        conn = psycopg2.connect(DATABASE_URL)
        cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT key, value FROM game_config;")
        rows = cur.fetchall()
        cur.close()
        conn.close()

        if not rows:
            return _env_fallback()

        db_cfg = {r["key"]: r["value"] for r in rows}

        int_fields = [
            "slots_total", "slots_per_person", "price_full", "price_half",
            "prize_1st", "prize_2nd", "prize_3rd", "winners_count",
            "warning_minutes", "low_slots_threshold",
        ]
        result = {}
        for k, default in _DEFAULTS.items():
            val = db_cfg.get(k, default)
            if k in int_fields and val is not None:
                result[k] = int(val)
            else:
                result[k] = val if val != "" else None

        return result

    except Exception:
        return _env_fallback()

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
