#!/usr/bin/env python3
"""
LOCOMO Benchmark for Nervon
============================
Uses Gemini Embedding 001 (key rotation across 3 projects) + Anthropic Haiku for QA.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import re
import string
import sys
import time
import logging
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, "/tmp/reasoning-memory")

from nervon.client import MemoryClient
import nervon.client as client_module
import nervon.pipeline.embeddings as embed_module
import nervon.retrieval.search as search_module
import litellm

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Load API keys
env_file = os.path.expanduser("~/.openclaw/secrets/openclaw.env")
if os.path.exists(env_file):
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))

# ---- Gemini Embedding with key rotation ----
GEMINI_KEYS = [
    "AIzaSyDrbuvkSkAv7rllnbK-3WVpfM99LrJ2mWg",  # Project: Nervon Benchmark
    "AIzaSyDfJBSqdisFwiXQBNQOOv-toL8g9Ry1lvU",  # Project: Nervon Benchmark 2
    "AIzaSyC6uEgXKiFRPzN1l9tgNJiF4ULX3LSrcpM",  # Project: Default (quota may be exhausted)
]
_key_index = 0
_call_count = 0

def _get_next_key():
    """Rotate keys every 50 calls to spread quota."""
    global _key_index, _call_count
    _call_count += 1
    if _call_count % 50 == 0:
        _key_index = (_key_index + 1) % len(GEMINI_KEYS)
    return GEMINI_KEYS[_key_index]

def gemini_get_embedding(text, model=None, task_type=None):
    import requests
    for attempt in range(len(GEMINI_KEYS) * 2):
        key = _get_next_key()
        try:
            r = requests.post(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent",
                params={"key": key},
                json={
                    "model": "models/gemini-embedding-001",
                    "content": {"parts": [{"text": text}]},
                    **({"taskType": task_type or "RETRIEVAL_DOCUMENT"} if task_type else {"taskType": "RETRIEVAL_DOCUMENT"}),
                },
                timeout=30,
            )
            if r.status_code == 200:
                return r.json()["embedding"]["values"]
            elif r.status_code == 429:
                global _key_index
                _key_index = (_key_index + 1) % len(GEMINI_KEYS)
                logger.info(f"Key {key[:10]}... rate limited, rotating to next key")
                time.sleep(1)
                continue
            else:
                logger.warning(f"Embedding API error {r.status_code}: {r.text[:100]}")
                time.sleep(2)
                continue
        except Exception as e:
            logger.warning(f"Embedding request failed: {e}")
            time.sleep(2)
            continue
    raise RuntimeError("All Gemini API keys exhausted")

def gemini_get_embeddings(texts, model=None, task_type=None):
    return [gemini_get_embedding(t, model, task_type) for t in texts]

# Monkey-patch all modules
embed_module.get_embedding = gemini_get_embedding
embed_module.get_embeddings = gemini_get_embeddings
client_module.get_embedding = gemini_get_embedding
client_module.get_embeddings = gemini_get_embeddings
search_module.get_embedding = gemini_get_embedding

# Set first key as default for any litellm calls
os.environ["GOOGLE_API_KEY"] = GEMINI_KEYS[0]

# Config
LLM_MODEL = "anthropic/claude-3-haiku-20240307"
EMBEDDING_MODEL = "gemini/gemini-embedding-001 (rotated)"
EMBEDDING_DIM = 3072
LOCOMO_PATH = "/tmp/locomo/data/locomo10.json"
RESULTS_DIR = "/tmp/reasoning-memory/benchmark_results"
DB_DIR = "/tmp/reasoning-memory/benchmark_dbs"

CATEGORY_NAMES = {
    1: "single-hop",
    2: "multi-hop",
    3: "temporal",
    4: "open-domain",
    5: "adversarial",
}

# ---- F1 Score (standard SQuAD-style) ----

def normalize_answer(s):
    s = str(s).lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = ' '.join(s.split())
    return s

def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return float(normalize_answer(prediction) == normalize_answer(ground_truth))
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)

# ---- Conversation extraction ----

def extract_conversations(sample):
    conv = sample["conversation"]
    sessions = []
    i = 1
    while f"session_{i}" in conv:
        session = conv[f"session_{i}"]
        date_time = conv.get(f"session_{i}_date_time", f"Session {i}")
        messages = []
        for turn in session:
            speaker = turn.get("speaker", turn.get("name", "unknown"))
            text = turn.get("text", turn.get("content", ""))
            if text:
                messages.append({
                    "role": "user" if speaker == conv.get("speaker_a", "A") else "assistant",
                    "content": f"[{date_time}] {speaker}: {text}"
                })
        if messages:
            sessions.append(messages)
        i += 1
    return sessions

# ---- QA with Nervon context ----

def answer_with_nervon(client, question, llm_model):
    context = client.get_context(question, max_tokens=3000)
    prompt = f"""Answer the question using ONLY the memory context below.
Give the shortest possible answer — ideally 1-5 words. No explanations, no sentences.
Use absolute dates (e.g., "June 2023", "March 18, 2026"), never relative dates (e.g., "yesterday", "last week").
If the answer is not in the context, reply exactly: "unanswerable"

Examples:
Q: What is John's job? A: Software engineer
Q: When did they meet? A: June 2023
Q: Where does she live? A: San Francisco
Q: What color is the car? A: unanswerable

MEMORY CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
    for attempt in range(5):
        try:
            response = litellm.completion(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                wait = min(2 ** attempt * 5, 60)
                logger.info(f"Rate limited on QA, waiting {wait}s (attempt {attempt+1}/5)")
                time.sleep(wait)
                continue
            logger.warning(f"LLM call failed: {e}")
            return "I don't know"
    return "I don't know"

# ---- Progress file (for external monitoring) ----

PROGRESS_FILE = os.path.join(RESULTS_DIR, "progress.json")

def save_progress(data):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(data, f, indent=2)

# ---- Main benchmark ----

def run_benchmark(max_samples=None, max_qa_per_sample=None):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DB_DIR, exist_ok=True)

    with open(LOCOMO_PATH) as f:
        data = json.load(f)

    if max_samples:
        data = data[:max_samples]

    all_results = []
    category_scores = defaultdict(list)
    overall_scores = []
    start_time = time.time()

    for sample_idx, sample in enumerate(data):
        sample_id = sample.get("sample_id", f"sample_{sample_idx}")
        sample_start = time.time()
        print(f"\n{'='*60}")
        print(f"[{sample_idx + 1}/{len(data)}] Processing: {sample_id}")
        print(f"{'='*60}")

        db_path = os.path.join(DB_DIR, f"{sample_id}.db")
        if os.path.exists(db_path):
            os.remove(db_path)

        client = MemoryClient(
            user_id="locomo",
            db_path=db_path,
            llm_model=LLM_MODEL,
            embedding_model=EMBEDDING_MODEL,
            embedding_dim=EMBEDDING_DIM,
        )

        # Step 1: Ingest
        sessions = extract_conversations(sample)
        n_memories = 0
        print(f"  📥 Ingesting {len(sessions)} sessions...")
        for si, session_msgs in enumerate(sessions):
            try:
                ids = client.add(session_msgs)
                n_memories += len(ids)
                if (si + 1) % 5 == 0:
                    print(f"    Session {si + 1}/{len(sessions)} done ({n_memories} memories so far)")
            except Exception as e:
                logger.warning(f"Failed to ingest session {si}: {e}")
        print(f"  ✅ Ingestion complete: {n_memories} memories stored")

        # Step 2: Answer questions
        qa_list = sample.get("qa", [])
        if max_qa_per_sample:
            qa_list = qa_list[:max_qa_per_sample]

        print(f"  📝 Answering {len(qa_list)} questions...")
        sample_results = []

        for qi, qa in enumerate(qa_list):
            question = qa["question"]
            ground_truth = qa.get("answer", qa.get("adversarial_answer", "unanswerable"))
            category = qa.get("category", 0)

            answer = answer_with_nervon(client, question, LLM_MODEL)
            score = f1_score(answer, ground_truth)
            time.sleep(0.5)  # Haiku rate limit buffer

            sample_results.append({
                "sample_id": sample_id,
                "question": question,
                "ground_truth": ground_truth,
                "prediction": answer,
                "f1": score,
                "category": category,
                "category_name": CATEGORY_NAMES.get(category, "unknown"),
            })
            category_scores[category].append(score)
            overall_scores.append(score)

            if (qi + 1) % 20 == 0:
                avg_so_far = sum(overall_scores) / len(overall_scores)
                print(f"    Q{qi + 1}/{len(qa_list)} done, running F1: {avg_so_far:.3f}")

        all_results.extend(sample_results)
        client.close()

        # Sample summary
        sample_avg = sum(r["f1"] for r in sample_results) / len(sample_results) if sample_results else 0
        sample_elapsed = time.time() - sample_start
        overall_avg = sum(overall_scores) / len(overall_scores) if overall_scores else 0

        # Per-category breakdown for this sample
        sample_cats = defaultdict(list)
        for r in sample_results:
            sample_cats[r["category"]].append(r["f1"])

        cat_line = " | ".join(
            f"{CATEGORY_NAMES.get(c, '?')}: {sum(s)/len(s):.2f}"
            for c, s in sorted(sample_cats.items())
        )

        progress = {
            "completed": sample_idx + 1,
            "total": len(data),
            "sample_id": sample_id,
            "sample_f1": round(sample_avg, 3),
            "overall_f1": round(overall_avg, 3),
            "memories_stored": n_memories,
            "questions_answered": len(qa_list),
            "sample_time_s": round(sample_elapsed, 1),
            "total_time_s": round(time.time() - start_time, 1),
            "category_breakdown": cat_line,
        }
        save_progress(progress)

        print(f"\n  📊 Sample {sample_idx + 1}/{len(data)} DONE")
        print(f"     F1: {sample_avg:.3f} | Overall: {overall_avg:.3f}")
        print(f"     {cat_line}")
        print(f"     ⏱️ {sample_elapsed:.0f}s | Memories: {n_memories} | QAs: {len(qa_list)}")

    # ---- Final report ----
    elapsed = time.time() - start_time
    overall_f1 = sum(overall_scores) / len(overall_scores) if overall_scores else 0

    print(f"\n{'='*60}")
    print(f"LOCOMO BENCHMARK RESULTS — Nervon v0.1.0")
    print(f"{'='*60}")
    print(f"LLM: {LLM_MODEL}")
    print(f"Embedding: {EMBEDDING_MODEL}")
    print(f"Samples: {len(data)}, Total QAs: {len(overall_scores)}")
    print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"\nOverall F1: {overall_f1:.3f}")
    print()
    print("Per-category F1:")
    cat_summary = {}
    for cat in sorted(category_scores.keys()):
        scores = category_scores[cat]
        avg = sum(scores) / len(scores)
        name = CATEGORY_NAMES.get(cat, f"cat_{cat}")
        print(f"  {name:15s}: {avg:.3f} ({len(scores)} questions)")
        cat_summary[name] = {"f1": round(avg, 3), "count": len(scores)}

    report = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "llm_model": LLM_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dim": EMBEDDING_DIM,
        },
        "overall_f1": round(overall_f1, 3),
        "category_scores": cat_summary,
        "total_questions": len(overall_scores),
        "elapsed_seconds": round(elapsed, 1),
        "details": all_results,
    }

    report_path = os.path.join(RESULTS_DIR, "locomo_results.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nDetailed results saved to: {report_path}")
    
    # Write final progress
    save_progress({
        "completed": len(data),
        "total": len(data),
        "status": "DONE",
        "overall_f1": round(overall_f1, 3),
        "category_scores": cat_summary,
        "total_time_s": round(elapsed, 1),
    })

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--qa-limit", type=int, default=None)
    args = parser.parse_args()
    run_benchmark(max_samples=args.samples, max_qa_per_sample=args.qa_limit)
