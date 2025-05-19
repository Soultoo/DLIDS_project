import numpy as np
from collections import Counter
import math

def compute_bleu_score(hypotheses, references, max_n=4):
    bleu_scores = []
    for hyp, refs in zip(hypotheses, references):
        bleu_scores.append(sentence_bleu(refs, hyp, max_n=max_n))
    return np.mean(bleu_scores)


def sentence_bleu(references, hypothesis, max_n=4):
    weights = [1.0 / max_n] * max_n
    bleu_score = 0.0
    for n in range(1, max_n + 1):
        bleu_score += weights[n - 1] * modified_precision(references, hypothesis, n)
    bleu_score *= brevity_penalty(references, hypothesis)
    return bleu_score


def modified_precision(references, hypothesis, n):
    hyp_ngrams = Counter(ngrams(hypothesis, n))
    max_ref_counts = Counter()
    for ref in references:
        ref_ngrams = Counter(ngrams(ref, n))
        for ngram in hyp_ngrams:
            max_ref_counts[ngram] = max(max_ref_counts.get(ngram, 0), ref_ngrams.get(ngram, 0))

    total_hyp_ngrams = sum(hyp_ngrams.values())
    if total_hyp_ngrams == 0:
        return 0.0

    clipped_counts = 0
    for ngram in hyp_ngrams:
        clipped_counts += min(hyp_ngrams[ngram], max_ref_counts[ngram])

    return clipped_counts / total_hyp_ngrams


def brevity_penalty(references, hypothesis):
    closest_ref_length = min(abs(len(hypothesis) - len(ref)) for ref in references)
    if closest_ref_length > 0:
        return math.exp(1 - len(references[0]) / len(hypothesis))
    else:
        return 1.0


def ngrams(sentence, n):
    return [tuple(sentence[i:i + n]) for i in range(len(sentence) - n + 1)]
