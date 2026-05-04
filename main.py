"""
main.py — Sentinel-DS CLI entrypoint.
Usage:
  python main.py --mode train --epochs 3
  python main.py --mode unlearn --record-id rec_0005
"""
import argparse
import json
import os

def load_records():
    path = "data/sample_records.json"
    if not os.path.exists(path):
        print("[main] sample_records.json not found. Running generate_data.py first...")
        import generate_data  # noqa: F401
    with open(path) as f:
        return json.load(f)

def mode_train(epochs: int):
    print(f"[main] Starting SISA training across 5 shards | epochs={epochs}")
    records = load_records()
    texts = [r["content"] for r in records]

    from src.llm_trainer import SISATrainer
    trainer = SISATrainer(num_shards=5, epochs=epochs)
    trainer.train(texts)
    print("[main] Training complete. Aggregated model saved to models/aggregated/")

def mode_unlearn(record_id: str):
    print(f"[main] Initiating surgical unlearning for record: {record_id}")
    records = load_records()
    texts = [r["content"] for r in records]
    ids = [r["id"] for r in records]

    if record_id not in ids:
        print(f"[main] ERROR: Record ID '{record_id}' not found in dataset.")
        return

    from src.unlearning_controller import UnlearningController
    controller = UnlearningController(num_shards=5, all_texts=texts, all_ids=ids)
    controller.unlearn(record_id)
    print(f"[main] Surgical unlearning complete for {record_id}.")

def main():
    parser = argparse.ArgumentParser(description="Sentinel-DS CLI")
    parser.add_argument("--mode", choices=["train", "unlearn"], required=True)
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs per shard")
    parser.add_argument("--record-id", type=str, default=None, help="Record ID to unlearn")
    args = parser.parse_args()

    if args.mode == "train":
        mode_train(args.epochs)
    elif args.mode == "unlearn":
        if not args.record_id:
            print("[main] ERROR: --record-id is required for unlearn mode.")
        else:
            mode_unlearn(args.record_id)

if __name__ == "__main__":
    main()