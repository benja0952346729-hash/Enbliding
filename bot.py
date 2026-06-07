import os
import re
import json
import logging
import psycopg2
import psycopg2.extras
from openai import OpenAI
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters
)
from config import get_game_config, get_api_key, rotate_key, DATABASE_URL, MODEL, BASE_URL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
ADMIN_IDS      = [int(x) for x in os.getenv("ADMIN_IDS", "").split(",") if x.strip()]

EMBED_MODEL = os.getenv("EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")
EMBED_DIM   = int(os.getenv("EMBED_DIM", "1024"))


# ══════════════════════════════════════════════════════════════════
# 1. DB CONNECTION
# ══════════════════════════════════════════════════════════════════

def get_conn():
    return psycopg2.connect(DATABASE_URL)


# ══════════════════════════════════════════════════════════════════
# 2. EMBEDDING
# ══════════════════════════════════════════════════════════════════

def get_embedding(text: str) -> list:
    while True:
        try:
            client = OpenAI(api_key=get_api_key(), base_url=BASE_URL)
            resp   = client.embeddings.create(model=EMBED_MODEL, input=text)
            return resp.data[0].embedding
        except Exception as e:
            err = str(e).lower()
            if "rate limit" in err or "429" in err or "quota" in err:
                rotate_key()
            else:
                logger.error(f"Embedding error: {e}")
                return []


# ══════════════════════════════════════════════════════════════════
# 3. VECTOR LOOKUP (አዲስ ⚡ — ፈጣን)
# ══════════════════════════════════════════════════════════════════

from trainer import extract_features

def vector_lookup(event_data: dict, threshold: float = 0.92) -> str | None:
    """
    feature vector → DB vector_store similarity search
    ቅርብ reply ካለ ይመልሳል፣ threshold ካልደረሰ None (→ AI fallback)
    """
    cfg      = get_game_config()
    features = extract_features(event_data, cfg)

    try:
        conn = get_conn()
        cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT
                reply,
                1 - (features <=> %s::vector) AS similarity
            FROM vector_store
            ORDER BY features <=> %s::vector
            LIMIT 1
        """, (features, features))
        row = cur.fetchone()
        cur.close()
        conn.close()

        if row and row["similarity"] >= threshold:
            return row["reply"]
        return None
    except Exception as e:
        logger.error(f"Vector lookup error: {e}")
        return None


# ══════════════════════════════════════════════════════════════════
# 4. RAG (edge case fallback)
# ══════════════════════════════════════════════════════════════════

def rag_search(user_message: str, limit: int = 6) -> list:
    embedding = get_embedding(user_message)
    if not embedding:
        return []
    try:
        conn = get_conn()
        cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT emb.event_type, emb.content, te.data,
                   1 - (emb.embedding <=> %s::vector) AS similarity
            FROM training_embeddings emb
            JOIN training_events te ON te.id = emb.event_id
            ORDER BY emb.embedding <=> %s::vector
            LIMIT %s
        """, (embedding, embedding, limit))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"RAG search error: {e}")
        return []

def format_rag_context(events: list) -> str:
    if not events:
        return ""
    lines = ["=== Training Examples ==="]
    for e in events:
        sim = e.get("similarity", 0)
        lines.append(f"[{e['event_type']}] (similarity: {sim:.2f})")
        lines.append(f"  {e.get('content', '')}")
        lines.append("")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# 5. AI CALL (edge case only)
# ══════════════════════════════════════════════════════════════════

def call_ai(messages: list, system_prompt: str = None) -> str:
    from config import API_KEYS
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + messages

    for attempt in range(max(len(API_KEYS), 1)):
        try:
            client = OpenAI(api_key=get_api_key(), base_url=BASE_URL)
            resp   = client.chat.completions.create(
                model=MODEL, messages=messages, max_tokens=1024,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err = str(e).lower()
            if "rate limit" in err or "429" in err or "quota" in err:
                rotate_key()
            else:
                raise e
    return "❌ AI አልተገናኘም። ቆይ ድጋሚ ሞክር።"


# ══════════════════════════════════════════════════════════════════
# 6. GAME STATE
# ══════════════════════════════════════════════════════════════════

from game_logic import Board, parse_request

active_boards:         dict[int, Board] = {}
board_message_ids:     dict[int, int]   = {}
remaining_message_ids: dict[int, int]   = {}

def get_board(chat_id: int) -> Board:
    if chat_id not in active_boards:
        active_boards[chat_id] = Board()
    return active_boards[chat_id]


# ══════════════════════════════════════════════════════════════════
# 7. DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════

def build_board_text(board: Board) -> str:
    cfg   = get_game_config()
    spp   = cfg["slots_per_person"]
    total = cfg["slots_total"]
    pf    = cfg["price_full"]
    ph    = cfg["price_half"]
    p1    = cfg["prize_1st"]
    p2    = cfg["prize_2nd"]
    p3    = cfg["prize_3rd"]
    users = total // spp

    lines = [
        f"በ {pf} ብር {spp} ቁጥሮችን በተከታታይ በመያዝ እድሎን ይሞክሩ "
        f"ለ {users} ሰው ብቻ ፈጣን ዕድል መልካም ዕድል\n",
        f"መደብ 👉በ {pf} ብር",
        f"       👉ግማሽ {ph} ብር\n",
        f"1ኛ 🥇{p1} ብር",
        f"2ኛ 🥈{p2}",
        f"3ኛ 🥉{p3}\n",
    ]

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
    if cfg.get("cbe_account"):
        lines.append(f"CBE {cfg['cbe_account']} {cfg.get('cbe_name','')}")
    if cfg.get("awash_account"):
        lines.append(f"አዋሽ  {cfg['awash_account']}")
    if cfg.get("dashen_account"):
        lines.append(f"ዳሽን  {cfg['dashen_account']}")
    if cfg.get("tele_birr"):
        lines.append(f"ቴሌ ብር {cfg['tele_birr']}")

    return "\n".join(lines)

def build_remaining_text(board: Board, keyword: str = "ቀሪ") -> str:
    cfg  = get_game_config()
    spp  = cfg["slots_per_person"]
    free = board.get_free_blocks(include_half=True)
    lines = [keyword]
    for b in free:
        if isinstance(b, int):
            start = (b - 1) * spp + 1
            lines.append(f"{start:02d}")
        else:
            b_num = int(str(b).replace("+", ""))
            start = (b_num - 1) * spp + 1
            lines.append(f"{start:02d}+")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# 8. AI DECIDE (edge case fallback — ሳይቀየር)
# ══════════════════════════════════════════════════════════════════

def ai_decide(chat_id: int, user_text: str, user_name: str, is_admin: bool) -> dict:
    board = get_board(chat_id)
    cfg   = get_game_config()

    events      = rag_search(user_text, limit=6)
    rag_ctx     = format_rag_context(events)
    free_blocks = board.get_free_blocks()
    taken       = cfg["slots_total"] // cfg["slots_per_person"] - len(free_blocks)

    system_prompt = f"""
You are a Telegram lottery bot assistant for an Ethiopian lottery game.
You understand both Amharic and English messages.

GAME STATE:
- Total blocks: {cfg['slots_total'] // cfg['slots_per_person']}
- Taken: {taken}
- Free: {len(free_blocks)}
- Price full: {cfg['price_full']} ETB
- Price half: {cfg['price_half']} ETB
- Is admin: {is_admin}
- User name: {user_name}

{rag_ctx}

INSTRUCTIONS:
- Analyze the user message and decide what action to take.
- Respond ONLY with a JSON object, no extra text.
- JSON format:
{{
  "action": "register" | "payment" | "show_board" | "show_remaining" | "winner" | "clear_slot" | "transfer" | "new_game" | "unknown",
  "reply": "bot reply in same language as user (Amharic or English)",
  "data": {{
    "block": <number or null>,
    "is_half": <true/false>,
    "name": "<name or null>",
    "amount": <number or null>,
    "from_block": <number or null>,
    "to_block": <number or null>
  }}
}}
"""

    messages = [{"role": "user", "content": user_text}]
    try:
        raw  = call_ai(messages, system_prompt)
        raw  = re.sub(r"```json|```", "", raw).strip()
        data = json.loads(raw)
        return data
    except Exception as e:
        logger.error(f"AI decide error: {e}")
        return {"action": "unknown", "reply": "ይቅርታ ልረዳ አልቻልኩም 🙏", "data": {}}


# ══════════════════════════════════════════════════════════════════
# 9. ACTION EXECUTOR (ሳይቀየር)
# ══════════════════════════════════════════════════════════════════

async def execute_action(update: Update, context: ContextTypes.DEFAULT_TYPE,
                         decision: dict, user_name: str):
    chat_id = update.effective_chat.id
    board   = get_board(chat_id)
    cfg     = get_game_config()
    action  = decision.get("action", "unknown")
    reply   = decision.get("reply", "ይቅርታ 🙏")
    data    = decision.get("data", {})

    async def update_board_remaining():
        free   = board.get_free_blocks()
        low_th = cfg["low_slots_threshold"]
        if len(free) == low_th:
            bm = await context.bot.send_message(chat_id, build_board_text(board))
            rm = await context.bot.send_message(chat_id, build_remaining_text(board))
            board_message_ids[chat_id]     = bm.message_id
            remaining_message_ids[chat_id] = rm.message_id
        elif len(free) < low_th:
            if chat_id in remaining_message_ids:
                try:
                    await context.bot.delete_message(chat_id, remaining_message_ids[chat_id])
                except Exception:
                    pass
            rm = await context.bot.send_message(chat_id, build_remaining_text(board))
            remaining_message_ids[chat_id] = rm.message_id
        if len(free) == 0:
            await context.bot.send_message(
                chat_id, build_board_text(board) + "\n\n🎰 ዕጣ ማውጫ ሰዓት ደረሰ! መልካም ዕድል 🙏")

    if action == "register":
        block   = data.get("block")
        is_half = data.get("is_half", False)
        name    = data.get("name") or user_name
        if block:
            success, reason = board.register(block, name, is_half)
            await update.message.reply_text(reply)
            if success:
                await update_board_remaining()
        else:
            await update.message.reply_text(reply)

    elif action == "payment":
        amount = data.get("amount", 0)
        if amount:
            board.apply_payment(user_name, amount)
        await update.message.reply_text(reply)

    elif action == "show_board":
        bm = await update.message.reply_text(build_board_text(board))
        board_message_ids[chat_id] = bm.message_id

    elif action == "show_remaining":
        await update.message.reply_text(build_remaining_text(board))

    elif action == "transfer":
        from_b = data.get("from_block")
        to_b   = data.get("to_block")
        if from_b and to_b:
            board.transfer(from_b, to_b)
        await update.message.reply_text(reply)

    elif action == "clear_slot":
        block = data.get("block")
        if block:
            start = board.get_block_start(block)
            for i in range(cfg["slots_per_person"]):
                board.slots[start + i].__init__(start + i)
        await update.message.reply_text(reply)

    elif action == "new_game":
        active_boards[chat_id] = Board()
        board_message_ids.pop(chat_id, None)
        remaining_message_ids.pop(chat_id, None)
        await update.message.reply_text(reply)
        await context.bot.send_message(chat_id, build_board_text(active_boards[chat_id]))

    elif action == "winner":
        import random
        taken   = [b for b in range(1, cfg["slots_total"] // cfg["slots_per_person"] + 1)
                   if not board.is_block_free(b)]
        w_count = min(cfg["winners_count"], len(taken))
        if w_count == 0:
            await update.message.reply_text("ያዘ የለም! 🙏")
            return
        w_blocks = random.sample(taken, w_count)
        prizes   = [cfg["prize_1st"], cfg["prize_2nd"], cfg["prize_3rd"]]
        medals   = ["🥇 1ኛ", "🥈 2ኛ", "🥉 3ኛ"]
        lines    = ["🎰 አሸናፊዎች:\n"]
        for rank, blk in enumerate(w_blocks):
            start = board.get_block_start(blk)
            name  = board.slots[start].name
            prize = prizes[rank] if rank < len(prizes) else 0
            lines.append(f"{medals[rank]}: {name} — {prize} ብር")
        await update.message.reply_text("\n".join(lines))

    else:
        await update.message.reply_text(reply)


# ══════════════════════════════════════════════════════════════════
# 10. TELEGRAM HANDLERS
# ══════════════════════════════════════════════════════════════════

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🎰 እንኳን ደህና መጡ!\n/board — board ይመልከቱ\n/remaining — ቀሪ slots"
    )

async def cmd_board(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    bm = await update.message.reply_text(build_board_text(get_board(chat_id)))
    board_message_ids[chat_id] = bm.message_id

async def cmd_remaining(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(build_remaining_text(get_board(update.effective_chat.id)))

async def cmd_new_game(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if update.effective_user.id not in ADMIN_IDS:
        await update.message.reply_text("❌ Admin ብቻ ነው!")
        return
    active_boards[chat_id] = Board()
    board_message_ids.pop(chat_id, None)
    remaining_message_ids.pop(chat_id, None)
    await update.message.reply_text("🎰 አዲስ ጨዋታ ተጀምሯል! መልካም ዕድል 🙏")
    await context.bot.send_message(chat_id, build_board_text(active_boards[chat_id]))

async def cmd_winner(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        await update.message.reply_text("❌ Admin ብቻ ነው!")
        return
    await execute_action(update, context, {"action": "winner", "reply": "", "data": {}}, "")

async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    chat_id   = update.effective_chat.id
    user      = update.effective_user
    user_name = user.first_name or user.username or str(user.id)
    is_admin  = user.id in ADMIN_IDS
    text      = update.message.text.strip()

    # ── Step 1: parse_request (ቁጥር ከሆነ — ፈጣን) ──────────────────
    parsed = parse_request(text)
    if parsed:
        board = get_board(chat_id)
        cfg   = get_game_config()

        for block, is_half, name_override in parsed:
            name            = name_override or user_name
            success, reason = board.register(block, name, is_half)
            free            = board.get_free_blocks()
            low_th          = cfg["low_slots_threshold"]

            if success:
                # ── Step 2: vector lookup ⚡ ──────────────────────
                event_data = {
                    "event_type": "registration",
                    "data": {
                        "block":            block,
                        "is_half":          is_half,
                        "partner":          None,
                        "remaining_blocks": len(free),
                        "lang":             "am",
                    }
                }
                reply = vector_lookup(event_data)

                # ── Step 3: fallback ──────────────────────────────
                if not reply:
                    reply = ("ጨዋታ ተሞልቷል 🙏" if len(free) == 0
                             else "እሺ ይፍጠን 🙏" if len(free) <= low_th
                             else "እሺ 🙏 ገቢ")
            else:
                # taken → vector lookup
                event_data = {
                    "event_type": "registration_failed",
                    "data": {"block": block, "is_half": is_half, "lang": "am"}
                }
                reply = vector_lookup(event_data) or "ተቀደምክ 🙏"

            await update.message.reply_text(reply)

            if success:
                if len(free) == low_th:
                    bm = await context.bot.send_message(chat_id, build_board_text(board))
                    rm = await context.bot.send_message(chat_id, build_remaining_text(board))
                    board_message_ids[chat_id]     = bm.message_id
                    remaining_message_ids[chat_id] = rm.message_id
                elif len(free) < low_th:
                    if chat_id in remaining_message_ids:
                        try:
                            await context.bot.delete_message(
                                chat_id, remaining_message_ids[chat_id])
                        except Exception:
                            pass
                    rm = await context.bot.send_message(chat_id, build_remaining_text(board))
                    remaining_message_ids[chat_id] = rm.message_id
                if len(free) == 0:
                    await context.bot.send_message(
                        chat_id,
                        build_board_text(board) + "\n\n🎰 ዕጣ ማውጫ ሰዓት ደረሰ! መልካም ዕድል 🙏")
        return

    # ── Step 4: AI (edge case) ────────────────────────────────────
    decision = ai_decide(chat_id, text, user_name, is_admin)
    await execute_action(update, context, decision, user_name)


# ══════════════════════════════════════════════════════════════════
# 11. MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    if not TELEGRAM_TOKEN:
        raise Exception("❌ TELEGRAM_TOKEN .env ላይ የለም!")

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start",     cmd_start))
    app.add_handler(CommandHandler("board",     cmd_board))
    app.add_handler(CommandHandler("remaining", cmd_remaining))
    app.add_handler(CommandHandler("newgame",   cmd_new_game))
    app.add_handler(CommandHandler("winner",    cmd_winner))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

    print("🤖 Bot ጀምሯል...")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
