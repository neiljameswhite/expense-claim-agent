#!/usr/bin/env python3
"""
Generate a receipt image for every corpus record.

    python scripts/make_receipts.py

Receipts are rendered *from* each record's extraction block, so the image and
the data the checks read cannot drift apart. Change the corpus and re-run.

Three cases are handled specially because they are the point of their record:

  is_receipt false      renders a boarding pass instead of a receipt
  low confidence        renders faded, blurred and skewed, so a reviewer can
                        see why extraction struggled
  no VAT number         omits the VAT block entirely rather than showing zero

Each receipt is the evidence for its own claim, so there is no watermark:
they are fixture data rather than placeholders.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus" / "corpus_v1.json"
OUT = ROOT / "ui" / "assets" / "receipts"

INK = "#2B2B2B"
FAINT = "#8A8A8A"
RULE = "#D8D4CC"

FONT_PATHS = {
    "mono": [
        "/System/Library/Fonts/Menlo.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ],
    "mono_bold": [
        "/System/Library/Fonts/Menlo.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    ],
    "sans": [
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ],
    "sans_bold": [
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ],
}


def font(kind: str, size: int):
    for path in FONT_PATHS[kind]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


# --------------------------------------------------------------------------
# Addresses, so each retailer looks like a real place rather than a name
# --------------------------------------------------------------------------

ADDRESSES = {
    "Pret A Manger, Derby": ["Unit 4, Intu Centre", "Derby  DE1 2PL"],
    "The Coach House, Bristol": ["14 Market Street", "Bristol  BS1 4QR"],
    "Ryman, Nottingham": ["22 Clumber Street", "Nottingham  NG1 3GA"],
    "Station Cafe, Sheffield": ["Sheffield Station, Concourse", "Sheffield  S1 2BP"],
    "LNER": ["Customer Services", "York  YO1 6JT"],
    "Independent Stationers, Huddersfield": ["8 Byram Street", "Huddersfield  HD1 1DR"],
    "Bella Italia, Manchester": ["The Printworks, Withy Grove", "Manchester  M4 2BS"],
    "The Bridgewater, Warrington": ["3 Bridge Street", "Warrington  WA1 2AB"],
    "The Ivy, Leeds": ["58 Vicar Lane", "Leeds  LS1 7JH"],
    "Hyatt Regency, Birmingham": ["2 Bridge Street", "Birmingham  B1 2JZ"],
    "British Airways": ["Waterside, PO Box 365", "Harmondsworth  UB7 0GB"],
    "Apex Grassmarket, Edinburgh": ["31-35 Grassmarket", "Edinburgh  EH1 2HS"],
    "Lufthansa": ["Deutsche Lufthansa AG", "60546 Frankfurt am Main"],
    "Institution of Engineering and Technology": ["Michael Faraday House", "Stevenage  SG1 2AY"],
    "Amazon UK": ["1 Principal Place, Worship Street", "London  EC2A 2FA"],
    "Wagamama, Warrington": ["Riverside Retail Park", "Warrington  WA2 7TT"],
    "Nuclear Institute": ["CK Hui Centre, 5 Tower Court", "Cambridge  CB3 0AX"],
    "Gaucho, Leeds": ["10 Bond Court", "Leeds  LS1 2JZ"],
    "Malmaison, Newcastle": ["104 Quayside", "Newcastle  NE1 3DX"],
    "Angelica, Leeds": ["Trinity Leeds, 5th Floor", "Leeds  LS1 6HW"],
    "Voco St Davids, Cardiff": ["Havannah Street", "Cardiff  CF10 5SD"],
}

FOOTERS = [
    "Thank you for your custom",
    "Please retain for your records",
    "VAT receipt — keep safe",
    "We hope to see you again",
    "Customer copy",
]


def address_for(retailer: str) -> list[str]:
    return ADDRESSES.get(retailer, ["", ""])


# --------------------------------------------------------------------------
# Finishing
# --------------------------------------------------------------------------


def finish(img: Image.Image) -> Image.Image:
    """A thin border so the receipt reads as a document against the page.

    No watermark: each receipt is the evidence for its own claim, generated
    from that record's extraction block. Marking them DEMO would imply they
    are placeholders standing in for something else, which they are not.
    """
    W, H = img.size
    ImageDraw.Draw(img).rectangle([(0, 0), (W - 1, H - 1)], outline="#E0DCD4", width=1)
    return img


# --------------------------------------------------------------------------
# Styles
# --------------------------------------------------------------------------


def thermal(rec: dict, seed: int) -> Image.Image:
    """Narrow monospace till roll. The default."""
    rng = random.Random(seed)
    e = rec["extraction"]
    items = e.get("line_items") or []

    W = 420
    H = 380 + 26 * len(items) + (90 if e.get("vat_number") else 0)
    img = Image.new("RGB", (W, H), "#FBFAF7")
    d = ImageDraw.Draw(img)

    big, mid, small = font("mono_bold", 19), font("mono", 14), font("mono", 11)
    y = 34

    retailer = (e.get("retailer") or "").upper()
    d.text((W // 2, y), retailer[:26], font=big, fill=INK, anchor="mt"); y += 30
    for line in address_for(e.get("retailer") or ""):
        if line:
            d.text((W // 2, y), line, font=small, fill=FAINT, anchor="mt"); y += 17
    if e.get("vat_number"):
        d.text((W // 2, y), f"VAT No. {e['vat_number']}", font=small, fill=FAINT, anchor="mt")
        y += 17
    y += 14

    d.line([(30, y), (W - 30, y)], fill=RULE); y += 22
    if e.get("date"):
        dt = e["date"]
        parts = dt.split("-")
        shown = f"{parts[2]}/{parts[1]}/{parts[0]}" if len(parts) == 3 else dt
        d.text((30, y), f"Date:  {shown}", font=mid, fill=INK); y += 20
    d.text((30, y), f"Time:  {rng.randint(7,21):02d}:{rng.randint(0,59):02d}", font=mid, fill=INK)
    y += 20
    d.text((30, y), f"Ref:   {rng.randint(100000, 999999)}", font=mid, fill=INK); y += 26

    d.line([(30, y), (W - 30, y)], fill=RULE); y += 22

    for item in items:
        name = str(item.get("description", ""))[:26]
        d.text((30, y), name, font=mid, fill=INK)
        d.text((W - 30, y), f"{float(item.get('cost', 0)):.2f}", font=mid, fill=INK, anchor="rt")
        y += 24

    y += 8
    d.line([(30, y), (W - 30, y)], fill=RULE); y += 20

    if e.get("vat_number") and e.get("vat_amount") is not None:
        net = float(e["total"]) - float(e["vat_amount"])
        d.text((30, y), "Subtotal", font=mid, fill=INK)
        d.text((W - 30, y), f"{net:.2f}", font=mid, fill=INK, anchor="rt"); y += 22
        d.text((30, y), "VAT @ 20%", font=mid, fill=INK)
        d.text((W - 30, y), f"{float(e['vat_amount']):.2f}", font=mid, fill=INK, anchor="rt")
        y += 26

    d.line([(30, y), (W - 30, y)], fill="#8A8A8A", width=2); y += 20
    d.text((30, y), "TOTAL", font=big, fill=INK)
    d.text((W - 30, y), f"{float(e['total']):.2f}", font=big, fill=INK, anchor="rt"); y += 40

    d.text((W // 2, y), f"PAID BY CARD  ****{rng.randint(1000,9999)}", font=small,
           fill=FAINT, anchor="mt"); y += 18
    d.text((W // 2, y), f"AUTH {rng.randint(100000,999999)}", font=small,
           fill=FAINT, anchor="mt"); y += 30
    d.text((W // 2, y), rng.choice(FOOTERS), font=small, fill=FAINT, anchor="mt")

    return img


def invoice(rec: dict, seed: int) -> Image.Image:
    """Wider printed invoice. For hotels, airlines and institutions."""
    rng = random.Random(seed)
    e = rec["extraction"]
    items = e.get("line_items") or []

    W = 480
    H = 360 + 28 * len(items) + (80 if e.get("vat_number") else 0)
    img = Image.new("RGB", (W, H), "#FFFFFF")
    d = ImageDraw.Draw(img)

    title, body, small = font("sans_bold", 20), font("sans", 13), font("sans", 11)
    y = 36

    d.text((36, y), e.get("retailer") or "", font=title, fill=INK); y += 28
    for line in address_for(e.get("retailer") or ""):
        if line:
            d.text((36, y), line, font=small, fill=FAINT); y += 16
    y += 10
    d.line([(36, y), (W - 36, y)], fill=RULE, width=1); y += 20

    d.text((36, y), "INVOICE", font=font("sans_bold", 13), fill=INK)
    d.text((W - 36, y), f"No. {rng.randint(10000,99999)}", font=body, fill=FAINT, anchor="rt")
    y += 24
    if e.get("date"):
        d.text((36, y), "Date", font=small, fill=FAINT)
        d.text((W - 36, y), e["date"], font=body, fill=INK, anchor="rt"); y += 22
    if e.get("vat_number"):
        d.text((36, y), "VAT registration", font=small, fill=FAINT)
        d.text((W - 36, y), e["vat_number"], font=body, fill=INK, anchor="rt"); y += 22

    y += 12
    d.line([(36, y), (W - 36, y)], fill=RULE); y += 18

    d.text((36, y), "Description", font=small, fill=FAINT)
    d.text((W - 36, y), "Amount", font=small, fill=FAINT, anchor="rt"); y += 20

    for item in items:
        d.text((36, y), str(item.get("description", ""))[:40], font=body, fill=INK)
        d.text((W - 36, y), f"{float(item.get('cost', 0)):.2f}", font=body, fill=INK, anchor="rt")
        y += 26

    y += 6
    d.line([(36, y), (W - 36, y)], fill=RULE); y += 20

    if e.get("vat_number") and e.get("vat_amount") is not None:
        net = float(e["total"]) - float(e["vat_amount"])
        d.text((36, y), "Net", font=body, fill=FAINT)
        d.text((W - 36, y), f"{net:.2f}", font=body, fill=INK, anchor="rt"); y += 22
        d.text((36, y), "VAT", font=body, fill=FAINT)
        d.text((W - 36, y), f"{float(e['vat_amount']):.2f}", font=body, fill=INK, anchor="rt")
        y += 26

    d.line([(36, y), (W - 36, y)], fill=INK, width=2); y += 18
    d.text((36, y), "Total due", font=font("sans_bold", 15), fill=INK)
    d.text((W - 36, y), f"GBP {float(e['total']):.2f}", font=font("sans_bold", 15),
           fill=INK, anchor="rt"); y += 40

    d.text((36, y), "Paid in full. Thank you.", font=small, fill=FAINT)
    return img


def boarding_pass(rec: dict, seed: int) -> Image.Image:
    """Not a receipt. Rendered for the record whose point is that it is not."""
    rng = random.Random(seed)
    W, H = 480, 300
    img = Image.new("RGB", (W, H), "#F4F7FA")
    d = ImageDraw.Draw(img)

    d.rectangle([(0, 0), (W, 54)], fill="#1F5FA8")
    d.text((24, 27), "BOARDING PASS", font=font("sans_bold", 17), fill="#FFFFFF",
           anchor="lm")
    d.text((W - 24, 27), "BA 1436", font=font("sans", 14), fill="#D6E4F5", anchor="rm")

    body, small = font("sans", 14), font("sans", 10)
    label = font("sans", 9)

    y = 82
    for x, lab, val in (
        (24, "PASSENGER", "OKAFOR / JAMES MR"),
        (24, "FROM", "MANCHESTER  MAN"),
        (250, "TO", "BREMEN  BRE"),
    ):
        pass

    d.text((24, y), "PASSENGER", font=label, fill=FAINT)
    d.text((24, y + 15), "OKAFOR / JAMES MR", font=body, fill=INK); y += 48

    d.text((24, y), "FROM", font=label, fill=FAINT)
    d.text((24, y + 15), "MANCHESTER  MAN", font=body, fill=INK)
    d.text((250, y), "TO", font=label, fill=FAINT)
    d.text((250, y + 15), "BREMEN  BRE", font=body, fill=INK); y += 48

    d.text((24, y), "DATE", font=label, fill=FAINT)
    d.text((24, y + 15), "25 AUG 2026", font=body, fill=INK)
    d.text((160, y), "GATE", font=label, fill=FAINT)
    d.text((160, y + 15), "B12", font=body, fill=INK)
    d.text((250, y), "SEAT", font=label, fill=FAINT)
    d.text((250, y + 15), "14C", font=body, fill=INK)
    d.text((350, y), "BOARDING", font=label, fill=FAINT)
    d.text((350, y + 15), "07:20", font=body, fill=INK); y += 52

    # barcode
    rng2 = random.Random(seed + 1)
    bx = 24
    while bx < W - 24:
        w = rng2.choice([2, 2, 3, 5])
        d.rectangle([(bx, y), (bx + w, y + 40)], fill=INK)
        bx += w + rng2.choice([2, 3, 4])

    d.text((24, y + 50), "This is not a receipt.", font=small, fill=FAINT)
    return img


def degrade(img: Image.Image, seed: int) -> Image.Image:
    """Make a receipt genuinely hard to read.

    Faded thermal print, a blur, a slight skew and some noise. The record
    this is applied to expects check 2 to fail, and a reviewer looking at
    the image should be able to see why.
    """
    rng = random.Random(seed)

    # fade towards white
    white = Image.new("RGB", img.size, "#FFFFFF")
    img = Image.blend(img, white, 0.55)

    img = img.filter(ImageFilter.GaussianBlur(radius=1.6))

    # skew
    W, H = img.size
    img = img.rotate(rng.uniform(-3.5, 3.5), resample=Image.BICUBIC,
                     fillcolor="#FFFFFF", expand=False)

    # a smudge across the lower half
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    top = int(H * 0.45)
    od.rectangle([(0, top), (W, top + int(H * 0.22))], fill=(255, 255, 255, 150))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    return img


# --------------------------------------------------------------------------

INVOICE_STYLE = {
    "Hyatt Regency, Birmingham",
    "Apex Grassmarket, Edinburgh",
    "Malmaison, Newcastle",
    "Voco St Davids, Cardiff",
    "British Airways",
    "Lufthansa",
    "LNER",
    "Institution of Engineering and Technology",
    "Nuclear Institute",
    "Amazon UK",
}


def build(rec: dict) -> Image.Image:
    e = rec["extraction"]
    seed = sum(ord(c) for c in rec["record_id"]) * 31

    if e.get("is_receipt") is False:
        return finish(boarding_pass(rec, seed))

    retailer = e.get("retailer")
    conf = e.get("confidence") or {}
    unreadable = not retailer or e.get("total") is None

    if unreadable:
        # Render plausible content, then destroy its legibility. The record
        # is about a receipt that exists but cannot be read.
        stand_in = {
            "extraction": {
                "is_receipt": True,
                "retailer": "The Cornerhouse, Preston",
                "date": rec["claim"]["claim_date"],
                "total": rec["claim"]["claim_amount"],
                "vat_number": None,
                "vat_amount": None,
                "line_items": [
                    {"description": "Lunch", "cost": rec["claim"]["claim_amount"]},
                ],
            }
        }
        return finish(degrade(thermal(stand_in, seed), seed))

    style = invoice if retailer in INVOICE_STYLE else thermal
    return finish(style(rec, seed))


def main() -> None:
    if not CORPUS.exists():
        raise SystemExit(f"Corpus not found at {CORPUS}")

    OUT.mkdir(parents=True, exist_ok=True)
    records = json.loads(CORPUS.read_text(encoding="utf-8"))["records"]

    for rec in records:
        img = build(rec)
        path = OUT / f"{rec['record_id']}.png"
        img.save(path)
        e = rec["extraction"]
        kind = (
            "boarding pass" if e.get("is_receipt") is False
            else "degraded" if not e.get("retailer") or e.get("total") is None
            else "invoice" if e.get("retailer") in INVOICE_STYLE
            else "till roll"
        )
        print(f"  {rec['record_id']:<4} {kind:<14} {img.size[0]}x{img.size[1]}  {path.name}")

    print(f"\n{len(records)} receipts written to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
