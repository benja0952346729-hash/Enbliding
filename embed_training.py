import os
import json
import time
import psycopg2
import psycopg2.extras
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

DATABASE_URL   = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL    = "text-embedding-3-small"
BATCH_SIZE     = 100  # OpenAI ላይ በ batch ይልካል

client = OpenAI(api_key=OPENAI_API_KEY)

# ══════════════════════════════════════════════════════════════════
# 1. DB SETUP — pgvector extension + table
# ══════════════════════════════════════════════════════════════════

def setup_vector_db():
    print("📦 pgvector setup...")
    conn = psycopg2.connect(DATABASE_URL)
    cur  = conn.cursor()

    # pgvector extension
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # Embeddings table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS training_embeddings (
            id         SERIAL PRIMARY KEY,
            event_id   INTEGER REFERENCES training_events(id),
            event_type TEXT,
            content    TEXT,
            embedding  vector(1536)
        );
    """)

    # Index — fast similarity search
    cur.execute("""
        CREATE INDEX IF NOT EXISTS embedding_idx
        ON training_embeddings
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("✅ pgvector ready!")


# ══════════════════════════════════════════════════════════════════
# 2. EVENT → TEXT — embedding ለማድረግ readable text
# ══════════════════════════════════════════════════════════════════

def event_to_text(event_type: str, data: dict) -> str:
    """
    Event data → single string ያደርጋል።
    AI ተምሮበት ያለውን format ይጠቀማል።
    """
    parts = [f"event: {event_type}"]

    # Registration
    if event_type == "registration":
        req    = data.get("user_request", "")
        name   = data.get("name", "")
        block  = data.get("block", "")
        half   = "ግማሽ" if data.get("is_half") else "ሙሉ"
        reply  = data.get("bot_reply", "")
        parts += [
            f"user: {req}",
            f"name: {name}",
            f"block: {block}",
            f"type: {half}",
            f"bot: {reply}",
        ]

    # Payment
    elif event_type == "payment":
        name   = data.get("name", "")
        amount = data.get("amount", "")
        reply  = data.get("bot_reply", "")
        parts += [
            f"name: {name}",
            f"amount: {amount}",
            f"bot: {reply}",
        ]

    # Unpaid warning
    elif event_type == "unpaid_warning":
        blocks  = data.get("unpaid_blocks", [])
        message = data.get("bot_message", "")
        parts  += [
            f"unpaid: {', '.join(blocks)}",
            f"bot: {message}",
        ]

    # Winner
    elif event_type == "winner":
        name   = data.get("name", "")
        rank   = data.get("rank", "")
        prize  = data.get("prize", "")
        msg    = data.get("bot_message", "")
        parts += [
            f"winner: {name}",
            f"rank: {rank}",
            f"prize: {prize}",
            f"bot: {msg}",
        ]

    # Board events
    elif event_type in ("board_with_remaining", "board_move", "all_paid_board"):
        action  = data.get("bot_action", "")
        trigger = data.get("trigger", "")
        parts  += [
            f"trigger: {trigger}",
            f"action: {action}",
        ]

    # Registration failed
    elif event_type == "registration_failed":
        block  = data.get("block", "")
        reason = data.get("reason", "")
        reply  = data.get("bot_reply", "")
        parts += [
            f"block: {block}",
            f"reason: {reason}",
            f"bot: {reply}",
        ]

    # New game
    elif event_type == "new_game":
        parts += [f"bot: {data.get('bot_message', '')}"]

    # Slot removed
    elif event_type == "slot_removed":
        parts += [
            f"block: {data.get('block', '')}",
            f"reason: {data.get('reason', '')}",
        ]

    # Winner balance
    elif event_type == "winner_balance":
        parts += [
            f"name: {data.get('name', '')}",
            f"prize: {data.get('prize', '')}",
            f"sent: {data.get('admin_sent', '')}",
            f"balance: {data.get('balance', '')}",
        ]

    # Fallback — ሁሉም keys
    else:
        for k, v in data.items():
            if isinstance(v, (str, int, float)):
                parts.append(f"{k}: {v}")

    return " | ".join(parts)


# ══════════════════════════════════════════════════════════════════
# 3. EMBED BATCH — OpenAI API
# ══════════════════════════════════════════════════════════════════

def embed_texts(texts: list) -> list:
    """texts list → embeddings list"""
    try:
        resp = client.embeddings.create(
            model=EMBED_MODEL,
            input=texts,
        )
        return [item.embedding for item in resp.data]
    except Exception as e:
        print(f"❌ Embedding error: {e}")
        time.sleep(5)
        return []


# ══════════════════════════════════════════════════════════════════
# 4. SAVE EMBEDDINGS → DB
# ══════════════════════════════════════════════════════════════════

def save_embeddings(rows: list):
    """rows: list of (event_id, event_type, content, embedding)"""
    if not rows:
        return
    conn = psycopg2.connect(DATABASE_URL)
    cur  = conn.cursor()
    psycopg2.extras.execute_values(
        cur,
        """
        INSERT INTO training_embeddings (event_id, event_type, content, embedding)
        VALUES %s
        ON CONFLICT DO NOTHING
        """,
        [(r[0], r[1], r[2], r[3]) for r in rows],
    )
    conn.commit()
    cur.close()
    conn.close()


# ══════════════════════════════════════════════════════════════════
# 5. MAIN — ሁሉም events embed ያደርጋል
# ══════════════════════════════════════════════════════════════════

def run_embedding():
    print("🚀 Embedding ጀምሯል...")
    setup_vector_db()

    conn = psycopg2.connect(DATABASE_URL)
    cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # Already embedded events ይዝለላል
    cur.execute("""
        SELECT te.id, te.event_type, te.data
        FROM training_events te
        LEFT JOIN training_embeddings emb ON te.id = emb.event_id
        WHERE emb.event_id IS NULL
        ORDER BY te.id
    """)
    events = cur.fetchall()
    cur.close()
    conn.close()

    total = len(events)
    if total == 0:
        print("✅ ሁሉም events already embedded ነው!")
        return

    print(f"📊 {total} events embed ያደርጋል...")

    done = 0
    for i in range(0, total, BATCH_SIZE):
        batch  = events[i: i + BATCH_SIZE]
        texts  = []
        meta   = []

        for e in batch:
            data    = e["data"] if isinstance(e["data"], dict) else json.loads(e["data"])
            content = event_to_text(e["event_type"], data)
            texts.append(content)
            meta.append((e["id"], e["event_type"], content))

        embeddings = embed_texts(texts)
        if not embeddings:
            print(f"⚠️  Batch {i} failed — skipping")
            continue

        rows = [
            (meta[j][0], meta[j][1], meta[j][2], embeddings[j])
            for j in range(len(embeddings))
        ]
        save_embeddings(rows)

        done += len(batch)
        pct   = int(done / total * 100)
        print(f"✅ {done}/{total} ({pct}%)", flush=True)

        # Rate limit avoid
        time.sleep(0.5)

    print(f"\n🎉 ተጠናቋል! {done} events → pgvector DB")


if __name__ == "__main__":
    run_embedding()
