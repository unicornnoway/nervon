#!/usr/bin/env python3
"""
LOCOMO Benchmark for Nervon
============================
Pipeline:
1. For each conversation in LOCOMO, feed all sessions into Nervon (add)
2. For each QA, use Nervon to retrieve context, then ask LLM to answer
3. Compare answer to ground truth using F1 score
4. Report per-category and overall scores
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

# Monkey-patch embeddings before importing nervon
from sentence_transformers import SentenceTransformer
_st_model = SentenceTransformer("all-MiniLM-L6-v2")

def _local_embed(text, model=None):
    return _st_model.encode(text).tolist()

import nervon.pipeline.embeddings as emb_mod
emb_mod.get_embedding = _local_embed
emb_mod.get_embeddings = lambda texts, model=None: [_local_embed(t) for t in texts]

from nervon.client import MemoryClient
from nervon import pipeline as pipeline_mod
from nervon.retrieval import search as search_mod

# Patch all import references
pipeline_mod.embeddings.get_embedding = _local_embed
search_mod.get_embedding = _local_embed

import litellm

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Load API key
env_file = os.path.expanduser("~/.openclaw/secrets/openclaw.env")
if os.path.exists(env_file):
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))

# Config
LLM_MODEL = "anthropic/claude-3-haiku-20240307"
EMBEDDING_MODEL = "local/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
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
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    # Remove articles
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    # Remove punctuation
    s = s.translate(str.maketrans('', '', string.punctuation))
    # Remove extra whitespace
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
    """Extract all sessions as message lists from a LOCOMO sample."""
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
    """Retrieve Nervon context and answer a question."""
    context = client.get_context(question, max_tokens=3000)

    prompt = f"""Based on the following memory context, answer the question concisely.
If the answer is not in the context, say "I don't know" or give your best guess.
Keep your answer brief — just the essential facts.

MEMORY CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    try:
        response = litellm.completion(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"LLM call failed: {e}")
        return "I don't know"


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
        print(f"\n{'='*60}")
        print(f"Processing sample {sample_idx + 1}/{len(data)}: {sample_id}")
        print(f"{'='*60}")

        # Create fresh DB for this sample
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

        # Step 1: Ingest all sessions
        sessions = extract_conversations(sample)
        print(f"  Ingesting {len(sessions)} sessions...")
        for si, session_msgs in enumerate(sessions):
            try:
                # Feed in batches of messages (each session as one add)
                client.add(session_msgs)
                if (si + 1) % 5 == 0:
                    print(f"    Session {si + 1}/{len(sessions)} done")
            except Exception as e:
                logger.warning(f"Failed to ingest session {si}: {e}")
        print(f"  Ingestion complete.")

        # Step 2: Answer QA questions
        qa_list = sample.get("qa", [])
        if max_qa_per_sample:
            qa_list = qa_list[:max_qa_per_sample]

        print(f"  Answering {len(qa_list)} questions...")
        sample_results = []

        for qi, qa in enumerate(qa_list):
            question = qa["question"]
            ground_truth = qa["answer"]
            category = qa.get("category", 0)

            answer = answer_with_nervon(client, question, LLM_MODEL)
            score = f1_score(answer, ground_truth)

            result = {
                "sample_id": sample_id,
                "question": question,
                "ground_truth": ground_truth,
                "prediction": answer,
                "f1": score,
                "category": category,
                "category_name": CATEGORY_NAMES.get(category, "unknown"),
            }
            sample_results.append(result)
            category_scores[category].append(score)
            overall_scores.append(score)

            if (qi + 1) % 20 == 0:
                avg_so_far = sum(overall_scores) / len(overall_scores)
                print(f"    Q{qi + 1}/{len(qa_list)} done, running F1: {avg_so_far:.3f}")

        all_results.extend(sample_results)
        client.close()

        # Sample summary
        sample_avg = sum(r["f1"] for r in sample_results) / len(sample_results) if sample_results else 0
        print(f"  Sample F1: {sample_avg:.3f}")

    # ---- Final report ----
    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"LOCOMO BENCHMARK RESULTS — Nervon v0.1.0")
    print(f"{'='*60}")
    print(f"LLM: {LLM_MODEL}")
    print(f"Embedding: {EMBEDDING_MODEL}")
    print(f"Samples: {len(data)}, Total QAs: {len(overall_scores)}")
    print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print()

    overall_f1 = sum(overall_scores) / len(overall_scores) if overall_scores else 0
    print(f"Overall F1: {overall_f1:.3f}")
    print()

    print("Per-category F1:")
    cat_summary = {}
    for cat in sorted(category_scores.keys()):
        scores = category_scores[cat]
        avg = sum(scores) / len(scores)
        name = CATEGORY_NAMES.get(cat, f"cat_{cat}")
        print(f"  {name:15s}: {avg:.3f} ({len(scores)} questions)")
        cat_summary[name] = {"f1": round(avg, 3), "count": len(scores)}

    # Save results
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

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=None, help="Max samples to process")
    parser.add_argument("--qa-limit", type=int, default=None, help="Max QAs per sample")
    args = parser.parse_args()

    run_benchmark(max_samples=args.samples, max_qa_per_sample=args.qa_limit)
