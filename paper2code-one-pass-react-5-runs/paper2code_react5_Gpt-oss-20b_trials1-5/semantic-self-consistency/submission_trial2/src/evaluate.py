from collections import Counter

def majority_vote(answers):
    """
    Traditional self‑consistency: majority vote on final answers.
    """
    counter = Counter(answers)
    return counter.most_common(1)[0][0]

def weighted_majority_vote(answers, weights):
    """
    Weighted majority vote using per‑sample weights.
    """
    score_by_answer = {}
    for ans, w in zip(answers, weights):
        score_by_answer[ans] = score_by_answer.get(ans, 0) + w
    return max(score_by_answer.items(), key=lambda x: x[1])[0]

def extract_answer(text):
    """
    Very light extraction: look for 'Answer:' prefix or fallback to last token.
    """
    if "Answer:" in text:
        return text.split("Answer:")[-1].strip().split("\n")[0]
    return text.strip().split()[-1]

def compute_accuracy(pred, true):
    if pred is None or true is None:
        return None
    return int(str(pred).strip().lower() == str(true).strip().lower())