import shutil
from pathlib import Path

SRC, DST = Path("food-101"), Path("data")
CLASSES = ["steak", "filet_mignon", "pork_chop", "prime_rib",        # hard: similar meats
           "sushi", "pizza", "ice_cream", "french_fries", "ramen", "caesar_salad"]

for split in ["train", "test"]:
    for line in (SRC / "meta" / f"{split}.txt").read_text().split():
        cls = line.split("/")[0]
        if cls in CLASSES:
            out = DST / split / cls
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy(SRC / "images" / f"{line}.jpg", out)

for split in ["train", "test"]:
    for c in CLASSES:
        print(split, c, len(list((DST / split / c).glob("*.jpg"))))