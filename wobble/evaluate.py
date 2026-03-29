"""WikiText-103 perplexity evaluation.

Sliding-window approach following the HuggingFace perplexity tutorial:
for each window of max_seq_length tokens, compute cross-entropy loss.
The stride parameter controls overlap between windows.
"""

import logging
import math
import time

import torch
from datasets import load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def evaluate_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int = 2048,
    stride: int = 512,
    max_tokens: int = 50_000,
) -> dict[str, float]:
    """Compute perplexity on WikiText-103 validation split.

    Uses a sliding window approach: the full validation text is tokenized
    into one long sequence, then evaluated in overlapping windows. Only
    the tokens in the non-overlapping portion of each window contribute
    to the final loss (except for the first window, where all tokens
    contribute).

    Args:
        model: Loaded model (already on device).
        tokenizer: Corresponding tokenizer.
        max_seq_length: Context window size for each forward pass.
        stride: Sliding window stride.
        max_tokens: Maximum total tokens to evaluate (0 = all).

    Returns:
        {'perplexity': float, 'loss': float, 'n_tokens': int}.
    """
    if stride > max_seq_length:
        raise ValueError(f"stride ({stride}) must be <= max_seq_length ({max_seq_length})")

    device = next(model.parameters()).device
    logger.info("Loading WikiText-103 validation split...")
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="validation")

    texts = [t for t in dataset["text"] if t.strip()]
    full_text = "\n\n".join(texts)
    logger.info("Concatenated %d non-empty texts, total %d chars", len(texts), len(full_text))

    encodings = tokenizer(
        full_text, return_tensors="pt", truncation=False, add_special_tokens=False,
    )
    input_ids = encodings["input_ids"]
    total_tokens = input_ids.size(1)
    if max_tokens > 0:
        total_tokens = min(total_tokens, max_tokens)
    logger.info("Evaluating %d tokens (max_seq_length=%d, stride=%d)", total_tokens, max_seq_length, stride)

    nlls: list[float] = []
    n_loss_tokens = 0
    last_log_time = time.monotonic()

    with torch.no_grad():
        for begin in range(0, total_tokens, stride):
            end = min(begin + max_seq_length, total_tokens)
            trg_len = end - begin if begin == 0 else min(stride, end - begin)

            window_ids = input_ids[:, begin:end].to(device)
            labels = window_ids.clone()
            context_len = window_ids.size(1) - trg_len
            if context_len > 0:
                labels[:, :context_len] = -100

            outputs = model(input_ids=window_ids, labels=labels)
            nlls.append(outputs.loss.float().item() * trg_len)
            n_loss_tokens += trg_len

            now = time.monotonic()
            if now - last_log_time >= 30.0:
                running_avg = sum(nlls) / n_loss_tokens
                logger.info(
                    "%d/%d tokens | running ppl: %.2f",
                    n_loss_tokens, total_tokens, math.exp(running_avg),
                )
                last_log_time = now

            if end >= total_tokens:
                break

    avg_loss = sum(nlls) / n_loss_tokens
    perplexity = math.exp(avg_loss)
    logger.info("Perplexity: %.4f (loss=%.4f, n_tokens=%d)", perplexity, avg_loss, n_loss_tokens)

    return {"perplexity": perplexity, "loss": avg_loss, "n_tokens": n_loss_tokens}
