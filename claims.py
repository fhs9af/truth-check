# claims.py
from models import nlp, flan_pipe

def extract_claims(text: str) -> list[str]:
    """
    1. Prompt Flan-T5 to list all factual claims (one per line).
    2. Split its output on newlines, strip blanks.
    3. Run each line through spaCy to split any compound sentences.
    """
    prompt = (
        "Extract all factual claims from the following text.\n"
        "Return one claim per line without numbering:\n\n"
        f"{text}\n\nClaims:"
    )
    output = flan_pipe(prompt, max_length=512, truncation=True)[0]["generated_text"]
    lines = [ln.strip() for ln in output.split("\n") if ln.strip()]

    claims: list[str] = []
    for ln in lines:
        doc = nlp(ln)
        for sent in doc.sents:
            claims.append(sent.text.strip())
    return claims
