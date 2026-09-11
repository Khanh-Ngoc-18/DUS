from datasets import load_dataset
import json
import os

def build_gsm8k(output_path="data/gsm8k/test.json"):
    ds = load_dataset("gsm8k", "main", split="test")

    samples = []
    for x in ds:
        answer = x["answer"].split("####")[-1].strip()
        samples.append({
            "question": x["question"],
            "answer": answer
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(samples)} samples to {output_path}")


if __name__ == "__main__":
    build_gsm8k()