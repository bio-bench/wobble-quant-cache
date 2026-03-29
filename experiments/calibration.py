"""Multi-domain calibration data loading with overlap verification.

No sample from the evaluation set may appear in the calibration set.
This is enforced by hash comparison before proceeding.
"""

import hashlib
import logging

import numpy as np
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class DataOverlapError(Exception):
    """Raised when calibration and evaluation data overlap is detected."""


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify_no_overlap(calibration_hashes: set[str], evaluation_hashes: set[str]) -> None:
    overlap = calibration_hashes & evaluation_hashes
    if overlap:
        raise DataOverlapError(
            f"FATAL: {len(overlap)} samples overlap between calibration and evaluation sets."
        )
    logger.info("Overlap check passed: 0 overlapping samples.")


def _collect_texts_to_budget(dataset_iter, tokenizer, target_tokens,
                             text_key="text", max_seq_length=2048):
    """Collect texts until token budget is met."""
    texts = []
    total_tokens = 0
    for sample in dataset_iter:
        text = sample[text_key]
        if not text or not text.strip():
            continue
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        for start in range(0, len(token_ids), max_seq_length):
            chunk_ids = token_ids[start : start + max_seq_length]
            chunk_text = tokenizer.decode(chunk_ids)
            texts.append(chunk_text)
            total_tokens += len(chunk_ids)
            if total_tokens >= target_tokens:
                return texts, total_tokens
    logger.warning("Dataset exhausted: %d / %d tokens", total_tokens, target_tokens)
    return texts, total_tokens


def load_calibration_data(
    tokenizer: PreTrainedTokenizerBase,
    wikitext_tokens: int = 10_000,
    c4_tokens: int = 10_000,
    pubmed_tokens: int = 5_000,
    max_seq_length: int = 2048,
    seed: int = 42,
) -> tuple[list[str], set[str]]:
    """Load multimodal calibration data with overlap verification.

    Sources: WikiText-103 train, C4, PubMed abstracts.

    Returns:
        (texts, text_hashes).
    """
    from datasets import load_dataset

    all_texts: list[str] = []

    # WikiText-103 train
    logger.info("Loading WikiText-103 train (%d tokens)...", wikitext_tokens)
    ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(ds))
    shuffled = ds.select(indices.tolist())
    texts, count = _collect_texts_to_budget(shuffled, tokenizer, wikitext_tokens,
                                            max_seq_length=max_seq_length)
    all_texts.extend(texts)
    logger.info("WikiText: %d tokens from %d texts", count, len(texts))

    # C4 (streaming)
    logger.info("Loading C4 sample (%d tokens)...", c4_tokens)
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True,
                      trust_remote_code=True)
    ds = ds.skip(seed * 1000)
    texts, count = _collect_texts_to_budget(ds, tokenizer, c4_tokens,
                                            max_seq_length=max_seq_length)
    all_texts.extend(texts)
    logger.info("C4: %d tokens from %d texts", count, len(texts))

    # PubMed abstracts
    logger.info("Loading PubMed abstracts (%d tokens)...", pubmed_tokens)
    try:
        ds = load_dataset("ccdv/pubmed-summarization", "document", split="train")
        text_key = "article"
    except Exception:
        logger.warning("PubMed dataset unavailable, using extra WikiText")
        ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
        indices = np.random.default_rng(seed + 7).permutation(len(ds))
        ds = ds.select(indices.tolist())
        text_key = "text"
    texts, count = _collect_texts_to_budget(ds, tokenizer, pubmed_tokens,
                                            text_key=text_key,
                                            max_seq_length=max_seq_length)
    all_texts.extend(texts)
    logger.info("PubMed/extra: %d tokens from %d texts", count, len(texts))

    text_hashes = {hash_text(t) for t in all_texts}

    # Verify no overlap with evaluation data
    logger.info("Loading evaluation hashes for overlap check...")
    eval_hashes: set[str] = set()
    wt_val = load_dataset("wikitext", "wikitext-103-v1", split="validation")
    for sample in wt_val:
        if sample["text"] and sample["text"].strip():
            eval_hashes.add(hash_text(sample["text"]))

    verify_no_overlap(text_hashes, eval_hashes)

    logger.info("Calibration data: %d texts, %d unique hashes", len(all_texts), len(text_hashes))
    return all_texts, text_hashes
