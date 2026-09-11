from datasets import load_dataset
import json
import os

def build_strategyqa(output_path="data/strategyqa/test.json"):
    print("Downloading StrategyQA...")

    ds = load_dataset("ChilleD/StrategyQA", split="test")

    samples = []
    for x in ds:
        samples.append({
            "question": x["question"],
            "answer": "yes" if x["answer"] else "no"
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(samples)} samples to {output_path}")


if __name__ == "__main__":
    build_strategyqa()