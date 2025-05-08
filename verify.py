# verify.py
import torch
import logging
from models import mnli_model, mnli_tokenizer

LABELS = ["Supported", "Contradicted", "Not Verifiable"]

def verify_claim(claim: str, evidence: str) -> dict:
    """
    1. If evidence=="No evidence found": short-circuit to Not Verifiable.
    2. Otherwise, for each LABELS entry:
       - Form hypothesis: "This claim is {LABEL} based on the evidence: {evidence}"
       - Tokenize (hypothesis, claim) as (premise, hypothesis) pair.
       - Run MNLI, softmax logits, take ENTAILMENT probability.
    3. Return the top label plus the full scores dict.
    """
    if evidence == "No evidence found":
        return {"label": "Not Verifiable", "scores": {}}

    try:
        # find which index = ENTAILMENT
        id2label   = mnli_model.config.id2label
        entail_idx = next(
            idx
            for idx,lab in id2label.items()
            if lab.upper() == "ENTAILMENT"
        )

        scores: dict[str,float] = {}
        for lab in LABELS:
            hypothesis = f"This claim is {lab} based on the evidence: {evidence[:512]}"
            # premise = hypothesis, hypothesis = claim
            inputs = mnli_tokenizer(
                hypothesis,
                claim,
                return_tensors="pt",
                truncation=True
            )
            outputs = mnli_model(**inputs)
            probs   = torch.softmax(outputs.logits, dim=1)
            scores[lab] = probs[0, entail_idx].item()

        best = max(scores, key=scores.get)
        return {"label": best, "scores": scores}

    except Exception:
        logging.exception(f"Verification error for claim: {claim}")
        return {"label": "Error", "scores": {}, "error": "See log for details"}
