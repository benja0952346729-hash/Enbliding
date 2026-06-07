import os
import re
import json
import logging
import psycopg2
import psycopg2.extras
from openai import OpenAI
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters
)
from config import get_game_config, get_api_key, rotate_key, DATABASE_URL, MODEL, BASE_URL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
ADMIN_IDS      = [int(x) for x in os.getenv("ADMIN_IDS", "").split(",") if x.strip()]
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL    = "text-embedding-3-small"

# OpenAI client — embedding ለብቻ
openai_client  = OpenAI(api_key=OPENAI_API_KEY)

# ══════════════════════════════════════════════════════════════════
# 1. DB CONNECTION
# ══════════════════════════════════════════════════════════════════

def get_conn():
    return psycopg2.connect(DATABASE_URL)


# ══════════════════════════════════════════════════════════════════
# 2. RAG — pgvector embedding search
# ══════════════════════════════════════════════════════════════════

def get_embedding(text: str) -> list:
    """User message → vector"""
    try:
        resp = openai_client.embeddings.create(
            model=EMBED_MODEL,
            input=text,
        )
        return resp.data[0].embedding
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return []


def rag_search(user_message: str, limit: int = 6) -> list:
    """
    pgvector cosine similarity search —
    user message ጋር ቅርብ training examples ያመጣል
    """
    embedding = get_embedding(user_message)
    if not embedding:
        return []

    try:
        conn = get_conn()
        cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Vector similarity search
        cur.execute("""
            SELECT
                emb.event_type,
                emb.content,
                te.data,
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
    """RAG examples → AI readable string"""
    if not events:
        return ""
    lines = ["=== Training Examples ==="]
    for e in events:
        sim = e.get("similarity", 0)
        lines.append(f"[{e['event_type']}] (similarity: {sim:.2f})")
        # content — already formatted text
        lines.append(f"  {e.get('content', '')}")
        lines.append("")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# 3. AI CALL — any model, auto rotate
# ══════════════════════════════════════════════════════════════════

def call_ai(messages: list, system_prompt: str = None) -> str:
    """Rate limit ሲመጣ auto rotate ያደርጋል"""
    from config import API_KEYS

    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + messages

    for attempt in range(max(len(API_KEYS), 1)):
        try:
            client = OpenAI(api_key=get_api_key(), base_url=BASE_URL)
            resp   = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=1024,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err = str(e).lower()
            if "rate limit" in err or "429" in err or "quota" in err:
                logger.warning(f"Rate limit — rotating key ({attempt+1})")
                rotate_key()
            else:
                raise e

    return "❌ AI አልተገናኘም። ቆይ ድጋሚ ሞክር።"


# ══════════════════════════════════════════════════════════════════
# 4. GAME STATE — active boards per chat
# ══════════════════════════════════════════════════════════════════

from game_logic import Board, parse_request

# chat_id → Board
active_boards: dict[int, Board] = {}
# chat_id → message_id of current board message
board_message_ids: dict[int, int] = {}
# chat_id → message_id of current remaining message
remaining_message_ids: dict[int, int] = {}


def get_board(chat_id: int) -> Board:
    if chat_id not in active_boards:
        active_boards[chat_id] = Board()
    return active_boards[chat_id]


# ══════════════════════════════════════════════════════════════════
# 5. DISPLAY HELPERS
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

    # Footer
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
    cfg        = get_game_config()
    spp        = cfg["slots_per_person"]
    free       = board.get_free_blocks(include_half=True)
    lines      = [keyword]
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
# 6. AI DECISION — ምን ማድረግ እንዳለበት ይወስናል
# ══════════════════════════════════════════════════════════════════

def ai_decide(chat_id: int, user_text: str, user_name: str, is_admin: bool) -> dict:
    """
    RAG + AI ተጠቅሞ action ይወስናል።
    Returns: {"action": ..., "reply": ..., "data": ...}
    """
    board = get_board(chat_id)
    cfg   = get_game_config()

    # ─ RAG examples ─
    events   = rag_search(user_text, limit=6)
    rag_ctx  = format_rag_context(events)

    # ─ Board state ─
    free_blocks  = board.get_free_blocks()
    taken_blocks = cfg["slots_total"] // cfg["slots_per_person"] - len(free_blocks)

    system_prompt = f"""
You are a Telegram lottery bot assistant for an Ethiopian lottery game.
You understand both Amharic and English messages.

GAME STATE:
- Total blocks: {cfg['slots_total'] // cfg['slots_per_person']}
- Taken: {taken_blocks}
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
- For registration: extract block number and half/full from message
- For payment: extract amount in ETB
- Reply must match training examples style
- Keep replies short like training data shows
"""

    messages = [{"role": "user", "content": user_text}]

    try:
        raw  = call_ai(messages, system_prompt)
        # JSON parse
        raw  = re.sub(r"```json|```", "", raw).strip()
        data = json.loads(raw)
        return data
    except Exception as e:
        logger.error(f"AI decide error: {e}")
        return {"action": "unknown", "reply": "ይቅርታ ልረዳ አልቻልኩም 🙏", "data": {}}


# ══════════════════════════════════════════════════════════════════
# 7. ACTION EXECUTOR — AI decision ይፈጽማል
# ══════════════════════════════════════════════════════════════════

async def execute_action(update: Update, context: ContextTypes.DEFAULT_TYPE,
                         decision: dict, user_name: str):
    chat_id = update.effective_chat.id
    board   = get_board(chat_id)
    cfg     = get_game_config()
    action  = decision.get("action", "unknown")
    reply   = decision.get("reply", "ይቅርታ 🙏")
    data    = decision.get("data", {})

    # ─ REGISTER ─────────────────────────────────────────────────
    if action == "register":
        block   = data.get("block")
        is_half = data.get("is_half", False)
        name    = data.get("name") or user_name

        if block:
            success, reason = board.register(block, name, is_half)
            if success:
                free   = board.get_free_blocks()
                low_th = cfg["low_slots_threshold"]

                await update.message.reply_text(reply)

                # Low slots → board + remaining
                if len(free) == low_th:
                    board_text = build_board_text(board)
                    rem_text   = build_remaining_text(board)
                    bm = await context.bot.send_message(chat_id, board_text)
                    rm = await context.bot.send_message(chat_id, rem_text)
                    board_message_ids[chat_id]     = bm.message_id
                    remaining_message_ids[chat_id] = rm.message_id

                # Already low → update remaining
                elif len(free) < low_th:
                    rem_text = build_remaining_text(board)
                    # Delete old remaining
                    if chat_id in remaining_message_ids:
                        try:
                            await context.bot.delete_message(
                                chat_id, remaining_message_ids[chat_id])
                        except Exception:
                            pass
                    rm = await context.bot.send_message(chat_id, rem_text)
                    remaining_message_ids[chat_id] = rm.message_id

                # Full → final board
                if len(free) == 0:
                    board_text = build_board_text(board)
                    await context.bot.send_message(
                        chat_id, board_text + "\n\n🎰 ዕጣ ማውጫ ሰዓት ደረሰ! መልካም ዕድል 🙏")
            else:
                await update.message.reply_text(reply)
        else:
            await update.message.reply_text(reply)

    # ─ PAYMENT ──────────────────────────────────────────────────
    elif action == "payment":
        amount = data.get("amount", 0)
        if amount:
            updated, remaining = board.apply_payment(user_name, amount)
            await update.message.reply_text(reply)
        else:
            await update.message.reply_text(reply)

    # ─ SHOW BOARD ───────────────────────────────────────────────
    elif action == "show_board":
        board_text = build_board_text(board)
        bm = await update.message.reply_text(board_text)
        board_message_ids[chat_id] = bm.message_id

    # ─ SHOW REMAINING ───────────────────────────────────────────
    elif action == "show_remaining":
        rem_text = build_remaining_text(board)
        await update.message.reply_text(rem_text)

    # ─ TRANSFER ─────────────────────────────────────────────────
    elif action == "transfer":
        from_b = data.get("from_block")
        to_b   = data.get("to_block")
        if from_b and to_b:
            success, msg = board.transfer(from_b, to_b)
            await update.message.reply_text(reply)
        else:
            await update.message.reply_text(reply)

    # ─ CLEAR SLOT ───────────────────────────────────────────────
    elif action == "clear_slot":
        block = data.get("block")
        if block:
            start = board.get_block_start(block)
            spp   = cfg["slots_per_person"]
            for i in range(spp):
                board.slots[start + i].__init__(start + i)
            await update.message.reply_text(reply)
        else:
            await update.message.reply_text(reply)

    # ─ NEW GAME ─────────────────────────────────────────────────
    elif action == "new_game":
        active_boards[chat_id]        = Board()
        board_message_ids.pop(chat_id, None)
        remaining_message_ids.pop(chat_id, None)
        board_text = build_board_text(active_boards[chat_id])
        await update.message.reply_text(reply)
        await context.bot.send_message(chat_id, board_text)

    # ─ WINNER ───────────────────────────────────────────────────
    elif action == "winner":
        import random
        taken   = [b for b in range(
            1, cfg["slots_total"] // cfg["slots_per_person"] + 1)
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

    # ─ UNKNOWN ──────────────────────────────────────────────────
    else:
        await update.message.reply_text(reply)


# ══════════════════════════════════════════════════════════════════
# 8. TELEGRAM HANDLERS
# ══════════════════════════════════════════════════════════════════

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cfg  = get_game_config()
    board = get_board(update.effective_chat.id)
    await update.message.reply_text(
        "🎰 እንኳን ደህና መጡ!\n/board — board ይመልከቱ\n/remaining — ቀሪ slots"
    )

async def cmd_board(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id    = update.effective_chat.id
    board      = get_board(chat_id)
    board_text = build_board_text(board)
    bm = await update.message.reply_text(board_text)
    board_message_ids[chat_id] = bm.message_id

async def cmd_remaining(update: Update, context: ContextTypes.DEFAULT_TYPE):
    board    = get_board(update.effective_chat.id)
    rem_text = build_remaining_text(board)
    await update.message.reply_text(rem_text)

async def cmd_new_game(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if update.effective_user.id not in ADMIN_IDS:
        await update.message.reply_text("❌ Admin ብቻ ነው!")
        return
    active_boards[chat_id]        = Board()
    board_message_ids.pop(chat_id, None)
    remaining_message_ids.pop(chat_id, None)
    board_text = build_board_text(active_boards[chat_id])
    await update.message.reply_text("🎰 አዲስ ጨዋታ ተጀምሯል! መልካም ዕድል 🙏")
    await context.bot.send_message(chat_id, board_text)

async def cmd_winner(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        await update.message.reply_text("❌ Admin ብቻ ነው!")
        return
    decision = {"action": "winner", "reply": "", "data": {}}
    await execute_action(update, context, decision, "")

async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """ሁሉም messages — AI ይወስናል"""
    if not update.message or not update.message.text:
        return

    chat_id   = update.effective_chat.id
    user      = update.effective_user
    user_name = user.first_name or user.username or str(user.id)
    is_admin  = user.id in ADMIN_IDS
    text      = update.message.text.strip()

    # ─ Quick parse — ቁጥር ብቻ ከሆነ directly register ─
    parsed = parse_request(text)
    if parsed:
        board = get_board(chat_id)
        cfg   = get_game_config()
        for block, is_half, name_override in parsed:
            name    = name_override or user_name
            success, reason = board.register(block, name, is_half)
            free    = board.get_free_blocks()
            low_th  = cfg["low_slots_threshold"]

            if success:
                if len(free) == 0:
                    reply = "ጨዋታ ተሞልቷል 🙏"
                elif len(free) <= low_th:
                    reply = "እሺ ይፍጠን 🙏"
                else:
                    reply = "እሺ 🙏 ገቢ"
            else:
                reply = "ተቀደምክ 🙏" if reason == "taken" else "ይቅርታ 🙏"

            await update.message.reply_text(reply)

            # Board/remaining logic
            if success:
                if len(free) == low_th:
                    board_text = build_board_text(board)
                    rem_text   = build_remaining_text(board)
                    bm = await context.bot.send_message(chat_id, board_text)
                    rm = await context.bot.send_message(chat_id, rem_text)
                    board_message_ids[chat_id]     = bm.message_id
                    remaining_message_ids[chat_id] = rm.message_id
                elif len(free) < low_th:
                    rem_text = build_remaining_text(board)
                    if chat_id in remaining_message_ids:
                        try:
                            await context.bot.delete_message(
                                chat_id, remaining_message_ids[chat_id])
                        except Exception:
                            pass
                    rm = await context.bot.send_message(chat_id, rem_text)
                    remaining_message_ids[chat_id] = rm.message_id
                if len(free) == 0:
                    board_text = build_board_text(board)
                    await context.bot.send_message(
                        chat_id,
                        board_text + "\n\n🎰 ዕጣ ማውጫ ሰዓት ደረሰ! መልካም ዕድል 🙏")
        return

    # ─ AI decision — ቁጥር ካልሆነ ─
    decision = ai_decide(chat_id, text, user_name, is_admin)
    await execute_action(update, context, decision, user_name)


# ══════════════════════════════════════════════════════════════════
# 9. MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    if not TELEGRAM_TOKEN:
        raise Exception("❌ TELEGRAM_TOKEN .env ላይ የለም!")

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start",     cmd_start))
    app.add_handler(CommandHandler("board",     cmd_board))
    app.add_handler(CommandHandler("remaining", cmd_remaining))
    app.add_handler(CommandHandler("newgame",   cmd_new_game))
    app.add_handler(CommandHandler("winner",    cmd_winner))

    # All messages → AI
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

    print("🤖 Bot ጀምሯል...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
