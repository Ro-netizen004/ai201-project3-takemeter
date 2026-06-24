"""Export a balanced IMDb sample to CSV for TakeMeter."""

from pathlib import Path

import pandas as pd
from datasets import load_dataset

LABEL_NAMES = {0: "negative", 1: "positive"}
SAMPLES_PER_CLASS = 150
OUTPUT = Path(__file__).resolve().parent.parent / "data" / "imdb_labeled.csv"


def main() -> None:
    ds = load_dataset("stanfordnlp/imdb", split="train")
    rows = []
    for label_id, label_name in LABEL_NAMES.items():
        subset = ds.filter(lambda x: x["label"] == label_id).shuffle(seed=42)
        for i in range(SAMPLES_PER_CLASS):
            rows.append(
                {
                    "text": subset[i]["text"],
                    "label": label_name,
                    "notes": "",
                }
            )

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    print(f"Saved {len(df)} examples to {OUTPUT}")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
