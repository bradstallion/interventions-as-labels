"""Extended capability battery (50 items): the off-target check for any
intervention (steering or weight edit). Scoring rule matches steer_eval:
answer must appear as substring of the lowercased reply — answers are chosen
to make substring collisions unlikely.

The original 12-item battery (M4 first pass) is the prefix; results stay
comparable by slicing."""

from __future__ import annotations

CAPABILITY_BATTERY: list[tuple[str, str]] = [
    # --- original 12 (M4 first pass) ---
    ("What is 17 + 26? Reply with the number only.", "43"),
    ("What is 9 * 8? Reply with the number only.", "72"),
    ("Spell the word 'cat' backwards. Reply with the letters only.", "tac"),
    ("Is 51 an odd number? Reply yes or no.", "yes"),
    ("What is the capital of France? One word.", "paris"),
    ("Complete: 2, 4, 8, 16, ... Reply with the next number only.", "32"),
    ("How many days are in a week? Number only.", "7"),
    ("Which is larger, 0.7 or 0.65? Reply with the number.", "0.7"),
    ("What is 100 - 37? Number only.", "63"),
    ("First letter of the alphabet? One letter.", "a"),
    ("Is the sun a star? Reply yes or no.", "yes"),
    ("What is 12 / 4? Number only.", "3"),
    # --- arithmetic & numbers ---
    ("What is 34 + 57? Number only.", "91"),
    ("What is 7 * 13? Number only.", "91"),
    ("What is 144 / 12? Number only.", "12"),
    ("What is 200 - 86? Number only.", "114"),
    ("What is 15% of 200? Number only.", "30"),
    ("Round 7.68 to one decimal place. Number only.", "7.7"),
    ("What is 2 to the power of 6? Number only.", "64"),
    ("How many minutes are in two hours? Number only.", "120"),
    ("What is the remainder of 17 divided by 5? Number only.", "2"),
    ("Complete: 1, 1, 2, 3, 5, 8, ... Next number only.", "13"),
    ("Complete: 100, 90, 80, ... Next number only.", "70"),
    ("Which is smaller, 1/4 or 1/3? Reply with the fraction.", "1/4"),
    # --- words & spelling ---
    ("Spell 'dog' backwards. Letters only.", "god"),
    ("Spell 'stop' backwards. Letters only.", "pots"),
    ("How many letters are in the word 'banana'? Number only.", "6"),
    ("What are the first two letters of 'water'? Two letters only.", "wa"),
    ("Which word is a fruit: chair, apple, brick? One word.", "apple"),
    ("Plural of 'child'? One word.", "children"),
    ("Past tense of 'go'? One word.", "went"),
    ("Opposite of 'cold'? One word.", "hot"),
    ("Spell 'sun' backwards. Letters only.", "nus"),
    ("Which is a color: seven, blue, run? One word.", "blue"),
    # --- facts ---
    ("What is the capital of Italy? One word.", "rome"),
    ("What is the capital of Japan? One word.", "tokyo"),
    ("How many continents are there? Number only.", "7"),
    ("How many legs does a spider have? Number only.", "8"),
    ("What planet do we live on? One word.", "earth"),
    ("Is ice colder than boiling water? Yes or no.", "yes"),
    ("Do fish live in water? Yes or no.", "yes"),
    ("Is 2 a prime number? Yes or no.", "yes"),
    ("How many sides does a hexagon have? Number only.", "6"),
    ("What gas do humans breathe in to survive? One word.", "oxygen"),
    # --- instruction following & format ---
    ("Reply with exactly the word OK.", "ok"),
    ("Write the number five as a digit.", "5"),
    ("List the first three letters of the alphabet separated by commas.", "a, b, c"),
    ("Repeat this word exactly: pineapple", "pineapple"),
    ("Answer in uppercase: what is the opposite of day? One word.", "night"),
    ("How many words are in this sentence: 'The cat sleeps'? Number only.", "3"),
]

assert len(CAPABILITY_BATTERY) == 50
