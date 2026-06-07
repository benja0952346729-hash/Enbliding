import random
import json
import psycopg2
import psycopg2.extras
from datetime import datetime, timedelta
from openai import OpenAI
from config import get_game_config, DATABASE_URL, get_api_key, rotate_key, BASE_URL
from game_logic import Board

# ── Embedding Client ─────────────────────────────────────────────
import os
EMBED_MODEL = os.getenv("EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")

def get_client():
    return OpenAI(
        api_key=get_api_key(),
        base_url=BASE_URL,
    )

def embed_texts(texts: list[str]) -> list[list[float]]:
    """texts → embeddings (rotation ይሠራል)"""
    is_nvidia = "nvidia" in BASE_URL or "nvidia" in EMBED_MODEL.lower()
    extra     = {"input_type": "passage", "truncate": "END"} if is_nvidia else {}

    while True:
        try:
            client = get_client()
            resp   = client.embeddings.create(
                model=EMBED_MODEL,
                input=texts,
                extra_body=extra if extra else None,
            )
            return [r.embedding for r in resp.data]
        except Exception as e:
            print(f"⚠️ Key error: {e} — rotating...")
            rotate_key()

cfg = get_game_config()

# ─── Sample Names ────────────────────────────────────────────────
AMHARIC_NAMES = [
    "አበበ", "አየለ", "ከበደ", "አልማዝ", "ሰላም", "ብርሃን", "ዮሃንስ", "ሄኖክ",
    "ናትናኤል", "ሚካኤል", "ሩት", "ማርያም", "ፍቅር", "ተወልደ", "ዳዊት", "ሳሙኤል",
    "እስቲፋኖስ", "ቤዛዊት", "ሙሉወርቅ", "ትዕግስት", "ፀጋዬ", "ገብሩ", "አስቴር",
    "ሕይወት", "ዘሪቱ", "ቃልኪዳን", "ኤፍሬም", "ቢኒያም", "ስምረት", "ዲና",
    "ፍሬሕይወት", "ወርቅነሽ", "ምህረት", "አሰፋ", "ግርማ", "ታደሰ", "ሙሉጌታ",
    "ሸዋዬ", "አስናቀ", "ደሳለኝ", "ጌታቸው", "አዱኛ", "ክብሮም", "ሃይሌ",
    "ልዕልት", "ትርሃስ", "አምሃ", "ጥላሁን", "ዘነበ", "ንጉሴ"
]
ENGLISH_NAMES = [
    "Abel", "Yonas", "Miki", "Sara", "Helen", "Biruk", "Nati", "Sam",
    "John", "Mary", "Alex", "Liya", "Eden", "Soli", "Bini", "Tina",
    "Roli", "Hana", "Dani", "Meron", "Sami", "Kidus", "Naol", "Femi",
    "Tsion", "Lidya", "Selam", "Ermias", "Fitsum", "Bereket"
]
ALL_NAMES = AMHARIC_NAMES + ENGLISH_NAMES

# ─── Request Styles ──────────────────────────────────────────────
HALF_KEYWORDS_AM = ["+", "÷", "ግ", "ግማሽ", "በግማሽ", "gmash", "gm", "g", "half"]
HALF_KEYWORDS_EN = ["+", "g", "gm", "gmash", "half"]

REGISTRATION_REPLIES_AM = [
    "እሺ 🙏 ገቢ", "ቤተሰብ ገቢ 🙏", "ገቢ እንዳይረሳ 🙏",
    "እሺ 🙏", "ገቢ 🙏", "ተቀበልን 🙏", "እሺ ይፍጠን 🙏",
]
REGISTRATION_REPLIES_EN = [
    "Done 🙏 registered", "Got it 🙏", "Registered 🙏",
    "OK 🙏", "Done 🙏",
]
TAKEN_REPLIES_AM = [
    "ተቀደምክ 🙏", "ተይዟል ይቅርታ 🙏", "ተቀድሟል 🙏",
    "ይህ ቁጥር ተወስዷል 🙏", "ቀድሞ ተወስዷል 🙏",
]
TAKEN_REPLIES_EN = [
    "Already taken 🙏", "Sorry, taken 🙏", "That number is taken 🙏",
]
REMAINING_KEYWORDS = ["ቀሪ", "ነቃይ", "remaining", "ቀሪዎች"]

# ─── Missing Patterns ────────────────────────────────────────────
REMAINING_QUESTIONS_AM = [
    "ቀሪ", "ቁጥር አለ?", "ምን ቀረ?", "ስንት ቀረ?", "ቁጥሮች አሉ?",
    "ነቃይ", "ቀሪዎች ስንት ናቸው?", "ምን አለ?", "ያለ ቁጥር?",
    "ቀሪ ቁጥር ስንት ነው?", "ስንት ቁጥር ቀረ?",
]
REMAINING_QUESTIONS_EN = [
    "remaining", "how many left?", "any slots?", "what's left?",
    "slots available?", "how many slots remain?", "left?",
]
REMAINING_REPLIES_AM = [
    "ቀሪ ቁጥሮች 👆", "እነዚህ ቀርተዋል 🙏", "ቀሪ 👆",
    "አሉ 🙏 ቀሪ ቁጥሮች 👆", "ይህን ይመልከቱ 👆",
]
REMAINING_REPLIES_EN = [
    "Remaining slots 👆", "These are left 🙏", "Available 👆",
]

BOARD_QUESTIONS_AM = [
    "board ይምጣ", "board አሳይ", "ሰንጠረዥ አሳይ",
    "ሁሉንም አሳይ", "board ልሳይ", "ሁኔታ ምን ይመስላል?",
]
BOARD_QUESTIONS_EN = [
    "show board", "board please", "show all", "what's the board?",
]
BOARD_REPLIES_AM = ["እነሆ 🙏 👆", "ይህ ነው board 🙏 👆", "👆 🙏"]
BOARD_REPLIES_EN = ["Here it is 🙏 👆", "Board 👆 🙏"]

TRANSFER_REQUESTS_AM = [
    "{f} ወደ {t} ቀይር", "{f} → {t}", "{f} ን {t} አድርገው",
    "{f} ቦታ {t} ሂድ", "{f} transfer {t}",
]
TRANSFER_REPLIES_AM = ["እሺ ቀየርኩ 🙏", "ተቀይሯል 🙏", "ቀይሬ ነው 🙏"]

CANCEL_REQUESTS_AM = [
    "{b} ሰርዝ", "{b} አስወግድ", "{b} ያዝ ሰርዝ", "cancel {b}",
]
CANCEL_REPLIES_AM  = ["ተሰርዟል 🙏", "እሺ ሰረዝኩ 🙏", "ተወግዷል 🙏"]

HAS_SLOTS_REPLIES_AM = [
    "አለ 🙏 ብዙ ቁጥሮች አሉ", "አሉ 🙏", "አለ ፈጠን 🙏",
    "አሁንም አለ 🙏 ፈጠን", "አለ ብዙ 🙏",
]
HAS_SLOTS_REPLIES_EN = [
    "Yes available 🙏", "Slots available 🙏", "Yes hurry 🙏",
]

# ─── Amharic Numbers ─────────────────────────────────────────────
AMHARIC_NUMBERS = {
    1: "አንድ", 2: "ሁለት", 3: "ሶስት", 4: "አራት", 5: "አምስት",
    6: "ስድስት", 7: "ሰባት", 8: "ስምንት", 9: "ዘጠኝ", 10: "አስር",
    11: "አስራ አንድ", 12: "አስራ ሁለት", 13: "አስራ ሶስት", 14: "አስራ አራት",
    15: "አስራ አምስት", 16: "አስራ ስድስት", 17: "አስራ ሰባት", 18: "አስራ ስምንት",
    19: "አስራ ዘጠኝ", 20: "ሃያ",
}

def random_half_keyword(lang="am"):
    return random.choice(HALF_KEYWORDS_EN if lang == "en" else HALF_KEYWORDS_AM)

def format_block_request(block, is_half, lang="am"):
    kw     = random_half_keyword(lang) if is_half else ""
    am_num = AMHARIC_NUMBERS.get(block, "")

    if lang == "en":
        styles = [
            f"{block:02d}{kw}", f"{block}{kw}",
            f"{block} take", f"give me {block}",
            f"block {block}", f"I want {block}",
            f"{block:02d} please", f"register {block}",
        ]
        if is_half:
            styles += [f"{block} half", f"{block:02d} half", f"{block} g", f"{block} gm"]
    else:
        styles = [
            f"{block:02d}{kw}", f"{block}{kw}",
            f"{block} ያዝ", f"{block:02d} ያዝ",
            f"{block} ይያዝ", f"{block:02d} ይያዝ",
            f"{block} ቁጥር ያዝ",
        ]
        if am_num:
            styles += [
                f"{am_num} ያዝ", f"{am_num}",
                f"{am_num} ቁጥር",
            ]
            if is_half:
                styles += [f"{am_num} ግማሽ", f"{am_num}+"]

        if is_half:
            styles += [
                f"{block} ግማሽ", f"{block:02d} ግማሽ",
                f"{block}+", f"{block:02d}+",
                f"{block} በግማሽ", f"{block} ÷",
            ]
    return random.choice(styles)

# ─── Board Display ───────────────────────────────────────────────
def get_block_start(block_number, slots_per_person):
    return (block_number - 1) * slots_per_person + 1

def build_board_header(cfg):
    spp   = cfg["slots_per_person"]
    total = cfg["slots_total"]
    users = total // spp
    pf    = cfg["price_full"]
    ph    = cfg["price_half"]
    p1    = cfg["prize_1st"]
    p2    = cfg["prize_2nd"]
    p3    = cfg["prize_3rd"]
    return (
        f"በ {pf} ብር {spp} ቁጥሮችን በተከታታይ በመያዝ እድሎን ይሞክሩ "
        f"ለ {users} ሰው ብቻ ፈጣን ዕድል መልካም ዕድል\n\n"
        f"መደብ 👉በ {pf} ብር \n"
        f"       👉ግማሽ {ph} ብር \n\n"
        f"1ኛ 🥇{p1} ብር \n2ኛ 🥈{p2}\n3ኛ 🥉{p3}\n"
    )

def build_board_footer(cfg):
    lines = []
    if cfg.get("cbe_account"):
        lines.append(f"CBE {cfg['cbe_account']} {cfg.get('cbe_name','')}")
    if cfg.get("awash_account"):
        lines.append(f"አዋሽ  {cfg['awash_account']}")
    if cfg.get("dashen_account"):
        lines.append(f"ዳሽን  {cfg['dashen_account']}")
    if cfg.get("tele_birr"):
        lines.append(f"ቴሌ ብር {cfg['tele_birr']}")
    return "\n".join(lines)

def display_board(board):
    cfg   = get_game_config()
    spp   = cfg["slots_per_person"]
    total = cfg["slots_total"]
    lines = [build_board_header(cfg)]
    for i in range(1, total + 1):
        slot        = board.slots[i]
        block_start = ((i - 1) // spp) * spp + 1
        if i == block_start and slot.name:
            mark     = "✅" if slot.paid_main  else ""
            reminder = "❓" if slot.reminder   else ""
            if slot.partner:
                pmark = "✅" if slot.paid_partner else ""
                lines.append(f"{i:02d}# {slot.name}{mark}{reminder}+ {slot.partner}{pmark}")
            elif slot.is_half:
                lines.append(f"{i:02d}# {slot.name}{mark}{reminder}+")
            else:
                lines.append(f"{i:02d}# {slot.name}{mark}{reminder}")
        else:
            lines.append(f"{i:02d}#")
        if i % spp == 0 and i < total:
            lines.append("")
    lines.append("")
    lines.append(build_board_footer(cfg))
    return "\n".join(lines)

def display_remaining(free_blocks, slots_per_person, keyword="ቀሪ"):
    lines = [keyword]
    for b in free_blocks:
        if isinstance(b, int):
            start = get_block_start(b, slots_per_person)
            lines.append(f"{start:02d}")
        else:
            b_num = int(str(b).replace("+", ""))
            start = get_block_start(b_num, slots_per_person)
            lines.append(f"{start:02d}+")
    return "\n".join(lines)

# ─── Event Content Formatter ─────────────────────────────────────
def format_event_content(event_type: str, data: dict) -> str:
    if event_type == "registration":
        return (
            f"user: {data.get('user_request', '')} "
            f"block: {data.get('block')} "
            f"name: {data.get('name', '')} "
            f"half: {data.get('is_half', False)} "
            f"reply: {data.get('bot_reply', '')}"
        )
    elif event_type == "registration_failed":
        return f"block: {data.get('block')} taken reply: {data.get('bot_reply', '')}"
    elif event_type == "payment":
        return (
            f"payment name: {data.get('name', '')} "
            f"amount: {data.get('amount', 0)} "
            f"reply: {data.get('bot_reply', '')}"
        )
    elif event_type == "unpaid_warning":
        return (
            f"unpaid warning blocks: {data.get('unpaid_blocks', [])} "
            f"message: {data.get('bot_message', '')}"
        )
    elif event_type == "winner":
        return (
            f"winner rank: {data.get('rank')} "
            f"name: {data.get('name', '')} "
            f"prize: {data.get('prize', 0)} "
            f"message: {data.get('bot_message', '')}"
        )
    elif event_type == "board_with_remaining":
        return f"board remaining trigger low slots free: {data.get('free_count')}"
    elif event_type == "new_game":
        return f"new game started: {data.get('bot_message', '')}"
    elif event_type == "remaining_query":
        return (
            f"user: {data.get('user_request', '')} "
            f"remaining: {data.get('remaining_blocks', 0)} "
            f"reply: {data.get('bot_reply', '')}"
        )
    elif event_type == "board_query":
        return f"user: {data.get('user_request', '')} reply: {data.get('bot_reply', '')}"
    elif event_type == "transfer":
        return (
            f"user: {data.get('user_request', '')} "
            f"from: {data.get('from_block')} to: {data.get('to_block')} "
            f"reply: {data.get('bot_reply', '')}"
        )
    elif event_type == "cancel":
        return (
            f"user: {data.get('user_request', '')} "
            f"block: {data.get('block')} "
            f"reply: {data.get('bot_reply', '')}"
        )
    else:
        return json.dumps(data, ensure_ascii=False)[:300]


# ══════════════════════════════════════════════════════════════════
# ── Feature Extraction (አዲስ) ─────────────────────────────────────
# ══════════════════════════════════════════════════════════════════

def extract_features(event: dict, cfg: dict) -> list[float]:
    """
    event → normalized feature vector
    [slot_norm, is_half, has_partner, remaining_norm, lang_am,
     is_taken, payment_norm, event_type_enc]
    """
    total_blocks = cfg["slots_total"] // cfg["slots_per_person"]
    data         = event.get("data", {})
    etype        = event.get("event_type", "")

    # slot normalized (0.0 - 1.0)
    block      = data.get("block") or 0
    slot_norm  = block / total_blocks if total_blocks > 0 else 0.0

    # is_half
    is_half    = 1.0 if data.get("is_half") else 0.0

    # has_partner
    has_partner = 1.0 if data.get("partner") else 0.0

    # remaining normalized
    remaining      = data.get("remaining_blocks", total_blocks)
    remaining_norm = remaining / total_blocks if total_blocks > 0 else 1.0

    # language (1=am, 0=en)
    lang_am = 1.0 if data.get("lang", "am") == "am" else 0.0

    # is_taken (registration_failed = taken)
    is_taken = 1.0 if etype == "registration_failed" else 0.0

    # payment normalized
    amount       = data.get("amount", 0) or 0
    price_full   = cfg["price_full"]
    payment_norm = min(amount / price_full, 1.0) if price_full > 0 else 0.0

    # event type encoding
    etype_map = {
        "registration":        0.1,
        "registration_failed": 0.2,
        "payment":             0.3,
        "unpaid_warning":      0.4,
        "winner":              0.5,
        "winner_balance":      0.6,
        "board_with_remaining":0.7,
        "remaining_update":    0.8,
        "board_move":          0.85,
        "all_paid_board":      0.9,
        "new_game":            1.0,
    }
    etype_enc = etype_map.get(etype, 0.0)

    return [
        slot_norm,
        is_half,
        has_partner,
        remaining_norm,
        lang_am,
        is_taken,
        payment_norm,
        etype_enc,
    ]


def get_reply_from_event(event: dict) -> str:
    """event → bot reply"""
    data  = event.get("data", {})
    etype = event.get("event_type", "")
    if etype in ("registration", "registration_failed", "payment"):
        return data.get("bot_reply", "")
    elif etype in ("unpaid_warning", "winner", "new_game",
                   "all_paid_board", "board_with_remaining"):
        return data.get("bot_message", "")
    elif etype in ("remaining_query", "board_query", "transfer", "cancel"):
        return data.get("bot_reply", "")


# ─── Simulate 1 Game ─────────────────────────────────────────────
def simulate_game(game_id):
    board  = Board()
    events = []
    cfg    = get_game_config()
    now    = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 365))
    spp    = cfg["slots_per_person"]

    total_blocks = cfg["slots_total"] // spp
    used_names   = []
    msg_count    = 0
    board_active = False

    def log(event_type, data):
        events.append({
            "game_id":    game_id,
            "event_type": event_type,
            "data":       data,
            "timestamp":  now.isoformat(),
            "content":    format_event_content(event_type, data),
        })

    blocks = list(range(1, total_blocks + 1))
    random.shuffle(blocks)

    for block in blocks:
        name    = random.choice(ALL_NAMES)
        lang    = "en" if name in ENGLISH_NAMES else "am"
        is_half = random.random() < 0.25

        if used_names and random.random() < 0.1:
            name = random.choice(used_names)

        partner = None
        if is_half and random.random() < 0.5:
            partner = random.choice(ALL_NAMES)

        success, reason = board.register(block, name, is_half, partner)

        if success:
            used_names.append(name)
            req         = format_block_request(block, is_half, lang)
            free_blocks = board.get_free_blocks()
            remaining   = len(free_blocks)

            if remaining == 0:
                bot_reply = "ጨዋታ ተሞልቷል 🙏" if lang == "am" else "Game is full 🙏"
            elif remaining <= cfg["low_slots_threshold"]:
                bot_reply = "እሺ ይፍጠን 🙏" if lang == "am" else "Hurry up! 🙏"
            else:
                bot_reply = (random.choice(REGISTRATION_REPLIES_AM) if lang == "am"
                             else random.choice(REGISTRATION_REPLIES_EN))

            log("registration", {
                "user_request": req, "block": block,
                "name": name, "is_half": is_half,
                "partner": partner, "bot_reply": bot_reply,
                "lang": lang, "remaining_blocks": remaining,
            })

            msg_count += 1
            keyword = random.choice(REMAINING_KEYWORDS)

            if remaining == cfg["low_slots_threshold"]:
                board_active = True
                msg_count    = 0
                log("board_with_remaining", {
                    "trigger":    "low_slots",
                    "board":      display_board(board),
                    "remaining":  display_remaining(free_blocks, spp, keyword),
                    "free_count": remaining,
                })
            elif board_active:
                log("remaining_update", {
                    "trigger":   "slot_taken",
                    "remaining": display_remaining(free_blocks, spp, keyword),
                })
                if msg_count >= 4:
                    msg_count = 0
                    log("board_move", {
                        "trigger":   "4_messages",
                        "board":     display_board(board),
                        "remaining": display_remaining(free_blocks, spp, keyword),
                    })
        else:
            reply = (random.choice(TAKEN_REPLIES_AM) if random.random() < 0.7
                     else random.choice(TAKEN_REPLIES_EN))
            log("registration_failed", {"block": block, "reason": reason, "bot_reply": reply})

        now += timedelta(minutes=random.randint(1, 10))

    # ── Missing Patterns — Remaining questions ───────────────────
    for _ in range(random.randint(2, 5)):
        lang      = random.choice(["am", "en"])
        free      = board.get_free_blocks()
        remaining = len(free)

        if lang == "am":
            user_req = random.choice(REMAINING_QUESTIONS_AM)
            if remaining > cfg["low_slots_threshold"]:
                bot_reply = random.choice(HAS_SLOTS_REPLIES_AM)
            else:
                bot_reply = random.choice(REMAINING_REPLIES_AM)
        else:
            user_req  = random.choice(REMAINING_QUESTIONS_EN)
            bot_reply = random.choice(REMAINING_REPLIES_EN) if remaining <= cfg["low_slots_threshold"] else random.choice(HAS_SLOTS_REPLIES_EN)

        log("remaining_query", {
            "user_request":     user_req,
            "bot_reply":        bot_reply,
            "remaining_blocks": remaining,
            "lang":             lang,
        })
        now += timedelta(minutes=random.randint(1, 5))

    # ── Missing Patterns — Board questions ───────────────────────
    for _ in range(random.randint(1, 3)):
        lang     = random.choice(["am", "en"])
        user_req = random.choice(BOARD_QUESTIONS_AM if lang == "am" else BOARD_QUESTIONS_EN)
        reply    = random.choice(BOARD_REPLIES_AM if lang == "am" else BOARD_REPLIES_EN)
        log("board_query", {
            "user_request": user_req,
            "bot_reply":    reply,
            "lang":         lang,
        })
        now += timedelta(minutes=random.randint(1, 5))

    # ── Missing Patterns — Transfer ──────────────────────────────
    taken_blocks = [b for b in range(1, total_blocks + 1) if not board.is_block_free(b)]
    free_blocks  = board.get_free_blocks()
    if taken_blocks and free_blocks and random.random() < 0.3:
        f        = random.choice(taken_blocks)
        t        = random.choice([b for b in free_blocks if isinstance(b, int)])
        user_req = random.choice(TRANSFER_REQUESTS_AM).format(f=f*cfg["slots_per_person"]-cfg["slots_per_person"]+1, t=t*cfg["slots_per_person"]-cfg["slots_per_person"]+1)
        reply    = random.choice(TRANSFER_REPLIES_AM)
        log("transfer", {
            "user_request": user_req,
            "bot_reply":    reply,
            "from_block":   f,
            "to_block":     t,
            "lang":         "am",
        })
        now += timedelta(minutes=random.randint(1, 5))

    # ── Missing Patterns — Cancel ─────────────────────────────────
    if taken_blocks and random.random() < 0.2:
        b        = random.choice(taken_blocks)
        slot_num = b * cfg["slots_per_person"] - cfg["slots_per_person"] + 1
        user_req = random.choice(CANCEL_REQUESTS_AM).format(b=slot_num)
        reply    = random.choice(CANCEL_REPLIES_AM)
        log("cancel", {
            "user_request": user_req,
            "bot_reply":    reply,
            "block":        b,
            "lang":         "am",
        })
        now += timedelta(minutes=random.randint(1, 5))
    for num, slot in board.slots.items():
        if not slot.is_taken:
            continue
        blk = (num - 1) // spp + 1
        if num != board.get_block_start(blk):
            continue
        if random.random() < 0.8:
            amount = cfg["price_half"] if slot.is_half else cfg["price_full"]
            updated, rem = board.apply_payment(slot.name, amount)
            log("payment", {
                "name": slot.name, "amount": amount,
                "updated_slots": updated, "remaining_balance": rem,
                "bot_reply": (f"{slot.name} ✅ ገቢ 🙏" if rem == 0
                              else f"{slot.name} {rem}ብር ቀርቷል ጨምር 🙏"),
            })
        now += timedelta(minutes=random.randint(1, 5))

    # Unpaid Warning
    unpaid = board.get_unpaid_blocks()
    if unpaid:
        log("unpaid_warning", {
            "unpaid_blocks": unpaid,
            "bot_message":   "⚠️ 2 ደቂቃ ይቀራል! ያልከፈሉ:\n" + "\n".join(unpaid),
        })

    # Final Board
    log("all_paid_board", {
        "board":       display_board(board),
        "bot_message": "🎰 ዕጣ ማውጫ ሰዓት ደረሰ! መልካም ዕድል 🙏",
    })

    # Winners
    taken    = [b for b in range(1, total_blocks + 1) if not board.is_block_free(b)]
    w_count  = min(cfg["winners_count"], len(taken))
    w_blocks = random.sample(taken, w_count)
    prizes   = [cfg["prize_1st"], cfg["prize_2nd"], cfg["prize_3rd"]]
    medals   = ["🥇 1ኛ", "🥈 2ኛ", "🥉 3ኛ"]
    w_names  = []

    for rank, blk in enumerate(w_blocks):
        start = board.get_block_start(blk)
        name  = board.slots[start].name
        prize = prizes[rank] if rank < len(prizes) else 0
        w_names.append(name)
        log("winner", {
            "rank": rank + 1, "block": blk, "name": name, "prize": prize,
            "bot_message": f"{medals[rank]}: {name} — {prize}ብር",
        })

    for rank, (blk, name) in enumerate(zip(w_blocks, w_names)):
        prize   = prizes[rank] if rank < len(prizes) else 0
        sent    = (random.randint(0, prize) // cfg["price_half"]) * cfg["price_half"]
        updated, removed, balance = board.apply_winner_balance(name, prize, sent)
        log("winner_balance", {
            "name": name, "prize": prize,
            "admin_sent": sent, "balance": balance,
            "auto_approved": updated, "auto_removed": removed,
            "admin_message": f"{rank + 1}={sent}",
        })

    log("new_game", {"bot_message": "🎰 አዲስ ጨዋታ ተጀምሯል! መልካም ዕድል 🙏"})

    return events


# ══════════════════════════════════════════════════════════════════
# ── Database ─────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════

def get_conn():
    return psycopg2.connect(DATABASE_URL)

def setup_db():
    print("📦 DB setup...")
    conn = get_conn()
    cur  = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS training_embeddings CASCADE;")
    cur.execute("DROP TABLE IF EXISTS training_events CASCADE;")
    cur.execute("DROP TABLE IF EXISTS vector_store CASCADE;")      # አዲስ
    print("🗑️ ያሉ tables ተሰርዘዋል")

    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # training_events
    cur.execute("""
        CREATE TABLE training_events (
            id         SERIAL PRIMARY KEY,
            game_id    INTEGER,
            event_type TEXT,
            data       JSONB,
            content    TEXT,
            timestamp  TIMESTAMP
        )
    """)

    # training_embeddings (RAG — እንደቀደመው)
    embed_dim = int(os.getenv("EMBED_DIM", "1024"))
    cur.execute(f"""
        CREATE TABLE training_embeddings (
            id         SERIAL PRIMARY KEY,
            event_id   INTEGER REFERENCES training_events(id),
            event_type TEXT,
            content    TEXT,
            embedding  vector({embed_dim})
        )
    """)
    cur.execute("""
        CREATE INDEX embedding_idx
        ON training_embeddings
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
    """)

    # ── vector_store (አዲስ — feature vectors) ────────────────────
    cur.execute("""
        CREATE TABLE vector_store (
            id         SERIAL PRIMARY KEY,
            event_id   INTEGER REFERENCES training_events(id),
            event_type TEXT,
            features   vector(8),
            reply      TEXT
        )
    """)
    cur.execute("""
        CREATE INDEX vector_store_idx
        ON vector_store
        USING ivfflat (features vector_cosine_ops)
        WITH (lists = 50);
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("✅ DB ready!")

def save_events(events):
    if not events:
        return []
    conn = get_conn()
    cur  = conn.cursor()
    rows = [
        (e["game_id"], e["event_type"],
         json.dumps(e["data"], ensure_ascii=False),
         e.get("content", ""), e["timestamp"])
        for e in events
    ]
    psycopg2.extras.execute_values(
        cur,
        "INSERT INTO training_events (game_id, event_type, data, content, timestamp) VALUES %s RETURNING id",
        rows
    )
    ids = [r[0] for r in cur.fetchall()]
    conn.commit()
    cur.close()
    conn.close()
    return ids

def save_embeddings(event_ids, events, embeddings):
    if not event_ids:
        return
    conn = get_conn()
    cur  = conn.cursor()
    rows = [
        (eid, event["event_type"], event.get("content", ""), emb)
        for eid, event, emb in zip(event_ids, events, embeddings)
    ]
    psycopg2.extras.execute_values(
        cur,
        "INSERT INTO training_embeddings (event_id, event_type, content, embedding) VALUES %s",
        rows
    )
    conn.commit()
    cur.close()
    conn.close()

def save_vectors(event_ids, events):
    """feature vectors → vector_store (አዲስ)"""
    if not event_ids:
        return
    cfg  = get_game_config()
    conn = get_conn()
    cur  = conn.cursor()
    rows = []
    for eid, event in zip(event_ids, events):
        features = extract_features(event, cfg)
        reply    = get_reply_from_event(event)
        if reply:  # reply ያለው ብቻ
            rows.append((eid, event["event_type"], features, reply))
    if rows:
        psycopg2.extras.execute_values(
            cur,
            "INSERT INTO vector_store (event_id, event_type, features, reply) VALUES %s",
            rows
        )
        conn.commit()
    cur.close()
    conn.close()


# ══════════════════════════════════════════════════════════════════
# ── Main ─────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════

def run_training(num_games=5000):
    print(f"🚀 Training ጀምሯል — {num_games} games...")
    setup_db()

    total_events = 0
    CHUNK        = 50

    for game_id in range(1, num_games + 1):
        print(f"🎮 Game {game_id} ጀምሯል...", flush=True)
        chunk_events = simulate_game(game_id)
        print(f"📝 {len(chunk_events)} events — DB saving...", flush=True)

        # 1. Save events
        event_ids = save_events(chunk_events)
        print(f"💾 Events saved — embedding...", flush=True)

        # 2. Embed (RAG)
        contents   = [e.get("content", "") for e in chunk_events]
        embeddings = embed_texts(contents)
        print(f"🔢 Embeddings done!", flush=True)

        # 3. Save embeddings (RAG)
        save_embeddings(event_ids, chunk_events, embeddings)

        # 4. Save feature vectors (አዲስ ⚡)
        save_vectors(event_ids, chunk_events)
        print(f"⚡ Feature vectors saved!", flush=True)

        total_events += len(chunk_events)

        if game_id % CHUNK == 0:
            pct = int(game_id / num_games * 100)
            print(f"✅ {game_id}/{num_games} ({pct}%) — {total_events} events", flush=True)

    print(f"\n🎉 ተጠናቋል!")
    print(f"   Games:  {num_games}")
    print(f"   Events: {total_events}")
    print(f"   ⏰ {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    run_training(5000)
