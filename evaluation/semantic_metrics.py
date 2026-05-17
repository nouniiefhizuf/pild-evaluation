"""Semantic quality metrics (ROUGE-L, BERTScore)."""

from typing import Dict
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import numpy as np


def compute_rouge_l(generated_code: str, reference_code: str) -> Dict:
    """Compute ROUGE-L score between generated and reference code.

    Uses longest common subsequence to capture structural similarity.
    """
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)

    scores = scorer.score(reference_code, generated_code)

    return {
        "rouge_l_precision": scores['rougeL'].precision,
        "rouge_l_recall": scores['rougeL'].recall,
        "rouge_l_f1": scores['rougeL'].fmeasure
    }


def compute_bertscore(generated_code: str, reference_code: str,
                      model_type: str = "microsoft/codebert-base",
                      device: str = "cuda") -> Dict:
    """Compute BERTScore using CodeBERT embeddings.

    Captures semantic equivalence beyond exact string matching.
    """
    # BERTScore expects lists
    cands = [generated_code]
    refs = [reference_code]

    P, R, F1 = bert_score(
        cands, refs,
        model_type=model_type,
        device=device,
        verbose=False
    )

    return {
        "bertscore_precision": P.mean().item(),
        "bertscore_recall": R.mean().item(),
        "bertscore_f1": F1.mean().item()
    }
