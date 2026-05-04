"""
src/llm_trainer.py — SISA Training Engine for DistilGPT-2.
Splits dataset into N shards, trains each in isolation, then aggregates weights.
CRITICAL: eval_strategy="no" must be used (not evaluation_strategy).
"""
import os
import json
import torch
import copy
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from datasets import Dataset

MODEL_NAME = "distilgpt2"
SHARD_DIR = "models/shards"
AGG_DIR = "models/aggregated"


class SISATrainer:
    def __init__(self, num_shards: int = 5, epochs: int = 1):
        self.num_shards = num_shards
        self.epochs = epochs
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        os.makedirs(SHARD_DIR, exist_ok=True)
        os.makedirs(AGG_DIR, exist_ok=True)

    def _tokenize(self, texts):
        def tokenize_fn(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=128,
                padding="max_length"
            )
        ds = Dataset.from_dict({"text": texts})
        return ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    def train_shard(self, shard_id: int, texts: list):
        print(f"  [SISA] Training shard {shard_id+1}/{self.num_shards} ({len(texts)} records)...")
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        tokenized = self._tokenize(texts)

        shard_path = os.path.join(SHARD_DIR, f"shard_{shard_id}")
        args = TrainingArguments(
            output_dir=shard_path,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=2,
            save_strategy="no",
            logging_steps=10,
            eval_strategy="no",   # CRITICAL: do NOT use evaluation_strategy
            report_to="none",
        )
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized,
            data_collator=data_collator,
        )
        trainer.train()
        model.save_pretrained(shard_path)
        self.tokenizer.save_pretrained(shard_path)
        print(f"  [SISA] Shard {shard_id+1} saved to {shard_path}")
        return model.state_dict()

    def aggregate(self, state_dicts: list):
        """Federated averaging of shard weights."""
        print("[SISA] Aggregating shard weights (FedAvg)...")
        avg_state = copy.deepcopy(state_dicts[0])
        for key in avg_state:
            for i in range(1, len(state_dicts)):
                avg_state[key] = avg_state[key] + state_dicts[i][key]
            avg_state[key] = avg_state[key] / len(state_dicts)

        agg_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        agg_model.load_state_dict(avg_state)
        agg_model.save_pretrained(AGG_DIR)
        self.tokenizer.save_pretrained(AGG_DIR)
        print(f"[SISA] Aggregated model saved to {AGG_DIR}")

    def train(self, texts: list):
        chunks = [texts[i::self.num_shards] for i in range(self.num_shards)]
        state_dicts = []
        for i, chunk in enumerate(chunks):
            if not chunk:
                continue
            sd = self.train_shard(i, chunk)
            state_dicts.append(sd)
        self.aggregate(state_dicts)

    def generate(self, prompt: str, max_new_tokens: int = 80) -> str:
        if not os.path.isdir(AGG_DIR):
            raise FileNotFoundError(f"Aggregated model not found at {AGG_DIR}")
        model = AutoModelForCausalLM.from_pretrained(AGG_DIR)
        tokenizer = AutoTokenizer.from_pretrained(AGG_DIR)
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id
            )
        return tokenizer.decode(output[0], skip_special_tokens=True)