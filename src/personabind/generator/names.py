from __future__ import annotations

import random
import string

NAME_STYLES: tuple[str, ...] = ("personal", "agentN", "letter")

PERSONAL_NAMES: tuple[str, ...] = (
    # ~220 given names; keep single-token. Trim/extend but keep >= 200 distinct.
    "Alice", "Bob", "Charles", "Doug", "Elena", "Farah", "Gustavo", "Hana",
    "Ingrid", "Jamal", "Keiko", "Liam", "Mei", "Nadia", "Omar", "Priya",
    "Quinn", "Rafael", "Sofia", "Tariq", "Ulla", "Viktor", "Wen", "Xavier",
    "Yara", "Zane", "Amara", "Bjorn", "Cyrus", "Dalia", "Eitan", "Fatima",
    "Goran", "Hina", "Isabel", "Junko", "Kwame", "Lucia", "Mateo", "Noor",
    "Otto", "Petra", "Rania", "Sven", "Tomas", "Uma", "Vera", "Wassim",
    "Ximena", "Yosef", "Zara", "Anton", "Bianca", "Caleb", "Dena", "Emil",
    "Freya", "Gisela", "Hugo", "Iris", "Joris", "Kira", "Leon", "Marta",
    "Nils", "Olga", "Pavel", "Rosa", "Said", "Tessa", "Ulf", "Vanya",
    "Willa", "Yannis", "Zeynep", "Aldo", "Berit", "Corin", "Dris", "Espen",
    "Fikret", "Greta", "Halvard", "Ilka", "Janek", "Karel", "Lasse", "Milena",
    "Nuno", "Oona", "Pia", "Rune", "Signe", "Timo", "Ursula", "Vidar",
    "Wilma", "Yusuf", "Zdenka", "Arto", "Bragi", "Cato", "Dagny", "Eero",
    "Folke", "Gita", "Hedda", "Ivo", "Jonna", "Kaisa", "Leif", "Maja",
    "Nestor", "Odd", "Palle", "Ragna", "Stig", "Torvald", "Unn", "Vigdis",
    "Wiebke", "Yngve", "Zsofia", "Ansel", "Bodil", "Csaba", "Dorka", "Enzo",
    "Fanni", "Gabor", "Henrik", "Ilona", "Janos", "Kata", "Lorand", "Marek",
    "Nandor", "Orsi", "Peti", "Reka", "Soma", "Tibor", "Ubul", "Vince",
    "Zita", "Arne", "Bea", "Cille", "Dag", "Eske", "Freddy", "Gorm",
    "Helle", "Ib", "Jeppe", "Karla", "Lone", "Mette", "Nanna", "Ole",
    "Pernille", "Rikke", "Soren", "Trine", "Ulrik", "Vibeke", "Yrsa", "Aksel",
    "Bent", "Carsten", "Dorte", "Egon", "Finn", "Gitte", "Hjalte", "Inger",
    "Jorn", "Kaj", "Lisbet", "Morten", "Niels", "Oluf", "Preben", "Rasmus",
    "Sanne", "Tage", "Ursel", "Verner", "Wanda", "Yvonne", "Zack", "Arden",
    "Blair", "Cody", "Devon", "Emery", "Frankie", "Gray", "Harper", "Indie",
    "Jules", "Kai", "Lane", "Marlow", "Nico", "Oakley", "Parker", "Reese",
    "Sasha", "Toby", "Val", "Wren", "Yael", "Zion", "Ada", "Basil",
)


def _letter_labels(n: int) -> list[str]:
    if n > 26:
        raise ValueError("letter style supports at most 26 agents")
    return [f"Agent {string.ascii_uppercase[i]}" for i in range(n)]


class NameSampler:
    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)
        self._bag: list[str] = []

    def pick_style(self, ratio: dict[str, float]) -> str:
        styles = list(NAME_STYLES)
        weights = [ratio[s] for s in styles]
        return self._rng.choices(styles, weights=weights, k=1)[0]

    def _refill_bag(self) -> None:
        pool = list(PERSONAL_NAMES)
        self._rng.shuffle(pool)
        self._bag = pool

    def draw(self, style: str, n: int) -> list[str]:
        if style == "agentN":
            return [f"Agent{i + 1}" for i in range(n)]
        if style == "letter":
            return _letter_labels(n)
        if style != "personal":
            raise ValueError(f"unknown name style {style!r}")
        out: list[str] = []
        while len(out) < n:
            if not self._bag:
                self._refill_bag()
            candidate = self._bag.pop()
            if candidate not in out:
                out.append(candidate)
        return out
