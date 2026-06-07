import random
import re
from config import get_game_config


# ─── Slot ───────────────────────────────────────────────────────
class Slot:
    def __init__(self, number):
        self.number       = number
        self.name         = None
        self.partner      = None
        self.is_half      = False
        self.paid_main    = False
        self.paid_partner = False
        self.reminder     = False

    @property
    def is_taken(self):
        return self.name is not None

    @property
    def is_free(self):
        return self.name is None

    def display(self):
        if self.is_free:
            return f"{self.number:02d}#"

        mark     = "✅" if self.paid_main  else ""
        reminder = "❓" if self.reminder   else ""

        if self.partner:
            pmark = "✅" if self.paid_partner else ""
            return f"{self.number:02d}# {self.name}{mark}{reminder}+ {self.partner}{pmark}"
        elif self.is_half:
            return f"{self.number:02d}# {self.name}{mark}{reminder}+"
        else:
            return f"{self.number:02d}# {self.name}{mark}{reminder}"


# ─── Board ──────────────────────────────────────────────────────
class Board:
    def __init__(self):
        cfg = get_game_config()
        self.slots_total      = cfg["slots_total"]
        self.slots_per_person = cfg["slots_per_person"]
        self.price_full       = cfg["price_full"]
        self.price_half       = cfg["price_half"]
        self.low_threshold    = cfg["low_slots_threshold"]
        self.slots            = {i: Slot(i) for i in range(1, self.slots_total + 1)}
        self.name_count       = {}

    # ── ስም ──────────────────────────────────────────────────────
    def resolve_name(self, raw_name):
        base = raw_name.strip()
        if base not in self.name_count:
            self.name_count[base] = 1
            return base
        else:
            self.name_count[base] += 1
            return f"{base} {self.name_count[base]}"

    # ── Block ─────────────────────────────────────────────────────
    def get_block_start(self, block_number):
        return (block_number - 1) * self.slots_per_person + 1

    def is_block_free(self, block_number):
        start = self.get_block_start(block_number)
        return all(self.slots[start + i].is_free for i in range(self.slots_per_person))

    def is_block_half_available(self, block_number):
        start = self.get_block_start(block_number)
        slot  = self.slots[start]
        return slot.is_taken and slot.is_half and slot.partner is None

    # ── Register ─────────────────────────────────────────────────
    def register(self, block_number, name, is_half=False, partner=None):
        total_blocks = self.slots_total // self.slots_per_person
        if not (1 <= block_number <= total_blocks):
            return False, "ቁጥር ልክ አይደለም"

        start         = self.get_block_start(block_number)
        resolved_name = self.resolve_name(name)

        if self.is_block_half_available(block_number):
            self.slots[start].partner = resolved_name
            return True, "partner"

        if not self.is_block_free(block_number):
            return False, "taken"

        for i in range(self.slots_per_person):
            s         = self.slots[start + i]
            s.name    = resolved_name
            s.is_half = is_half
            if partner:
                s.partner = partner

        return True, "registered"

    # ── Transfer ──────────────────────────────────────────────────
    def transfer(self, from_block, to_block):
        if not self.is_block_free(to_block):
            return False, "to block ተይዟል"

        from_start = self.get_block_start(from_block)
        to_start   = self.get_block_start(to_block)

        for i in range(self.slots_per_person):
            t              = self.slots[to_start   + i]
            f              = self.slots[from_start + i]
            t.name         = f.name
            t.partner      = f.partner
            t.is_half      = f.is_half
            t.paid_main    = f.paid_main
            t.paid_partner = f.paid_partner
            t.reminder     = f.reminder
            f.__init__(f.number)

        return True, "transferred"

    # ── Payment ──────────────────────────────────────────────────
    def apply_payment(self, name, amount):
        remaining = amount
        updated   = []

        for slot in self.slots.values():
            if remaining <= 0:
                break
            if not slot.is_taken:
                continue
            if slot.name != name and slot.partner != name:
                continue

            if not slot.paid_main:
                cost = self.price_half if slot.is_half else self.price_full
                if remaining >= cost:
                    slot.paid_main = True
                    remaining     -= cost
                    updated.append(slot.number)
                    slot.reminder  = False
                else:
                    if remaining == self.price_half and not slot.is_half:
                        slot.reminder = True
                    break

            elif slot.partner and not slot.paid_partner:
                if remaining >= self.price_half:
                    slot.paid_partner = True
                    remaining        -= self.price_half
                    updated.append(slot.number)

        return updated, remaining

    # ── Unpaid Warning ────────────────────────────────────────────
    def get_unpaid_blocks(self):
        unpaid = []
        seen   = set()

        for num, slot in self.slots.items():
            if not slot.is_taken:
                continue
            block = (num - 1) // self.slots_per_person + 1
            if block in seen:
                continue
            seen.add(block)

            main_unpaid    = not slot.paid_main
            partner_unpaid = slot.partner and not slot.paid_partner

            if main_unpaid and partner_unpaid:
                unpaid.append(f"{block:02d}")
            elif main_unpaid or partner_unpaid:
                unpaid.append(f"{block:02d}+")

        return unpaid

    # ── Free Slots ───────────────────────────────────────────────
    def get_free_blocks(self, include_half=False):
        free         = []
        total_blocks = self.slots_total // self.slots_per_person

        for b in range(1, total_blocks + 1):
            if self.is_block_free(b):
                free.append(b)
            elif include_half and self.is_block_half_available(b):
                free.append(f"{b}+")

        return free

    # ── Winner Balance ────────────────────────────────────────────
    def apply_winner_balance(self, winner_name, prize, sent_amount):
        balance = prize - sent_amount
        covered = 0
        updated = []
        removed = []

        for slot in self.slots.values():
            if slot.name != winner_name and slot.partner != winner_name:
                continue

            cost = self.price_half if slot.is_half else self.price_full

            if covered + cost <= balance:
                slot.paid_main = True
                covered       += cost
                updated.append(slot.number)
            else:
                if slot.paid_main:
                    slot.paid_main = False
                    removed.append(slot.number)

        return updated, removed, balance

    # ── Display ───────────────────────────────────────────────────
    def display(self):
        lines = []
        for num in range(1, self.slots_total + 1):
            slot        = self.slots[num]
            block_start = self.get_block_start(
                (num - 1) // self.slots_per_person + 1
            )
            if num == block_start:
                lines.append(slot.display())
            else:
                lines.append(f"{num:02d}#")
        return "\n".join(lines)


# ─── Helper: Parse Request ────────────────────────────────────────
def parse_request(text):
    text = text.strip()
    text = re.sub(r"[,/\n]+", " ", text)

    global_half_pattern = r"(ሁሉንም\s*(በግማሽ|\+)|all\s*half)"
    global_full_pattern = r"(ሁሉንም\s*(በሙሉ|ሙሉ)|all\s*full|bemulu)"
    is_global_half      = bool(re.search(global_half_pattern, text, re.IGNORECASE))
    is_global_full      = bool(re.search(global_full_pattern, text, re.IGNORECASE))

    half_kw = r"(\+|÷|ግ\b|ግማሽ|በግማሽ|gmash|gm\b|g\b|half)"

    name_match    = re.search(
        r"(\d+)\s+([^\d\+÷]+?)\s*(በል|ብለህ\s*ያዝ|register|bel|ble)",
        text, re.IGNORECASE
    )
    name_override = name_match.group(2).strip() if name_match else None

    tokens  = text.split()
    results = []

    for token in tokens:
        num_match = re.match(
            r"^(\d+)(" + half_kw[1:-1] + r")?$", token, re.IGNORECASE
        )
        if num_match:
            block   = int(num_match.group(1))
            is_half = bool(num_match.group(2)) or is_global_half
            if is_global_full:
                is_half = False
            results.append((block, is_half, name_override))

    return results
