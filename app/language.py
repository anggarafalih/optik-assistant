"""Detect whether a question is in Bahasa Indonesia or English.

Small local models often answer in the language of the sources instead of the question.
Detecting the language in code and naming it explicitly in the prompt is more reliable
than asking the model to work it out. The questions only ever come in two languages, so
counting common function words is enough; no language-ID library is needed.
"""

import re
from typing import Literal

Language = Literal["id", "en"]

_INDONESIAN = set("""
yang apa apakah bagaimana gimana berapa bisa bisakah boleh bolehkah saya aku kami kita anda kamu
di ke dari dan atau untuk dengan ini itu tidak tak gak nggak ga ada adalah jika kalau kapan
mengapa kenapa dimana mana sudah belum perlu harus mau ingin cara arti lama pakai memakai
kacamata lensa mata resep toko harga jam buka hari dong sih ya juga saja apakah bagi
""".split())

_ENGLISH = set("""
the a an is are was do does did can could should would will what how why when where which who
my i me you your it its of to for with and or in on at be have has much many long this that
glasses lens lenses eye eyes store price
""".split())


def detect_language(text: str) -> Language:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    indonesian = sum(w in _INDONESIAN for w in words)
    english = sum(w in _ENGLISH for w in words)
    # Ties (e.g. a bare product name) default to Indonesian, the store's main language.
    return "en" if english > indonesian else "id"
