import numpy as np
from collections import Counter
import math
from torchtext.data.metrics import bleu_score

def compute_bleu_score(hypotheses, references, max_n=4):
    bleu_scores = []
    for hyp, refs in zip(hypotheses, references):
        bleu_scores.append(sentence_bleu(refs, hyp, max_n=max_n))
    return np.mean(bleu_scores)

# Pauls implementation used a linear average the original paper and standard version of BLEU
# uses gemetric mean instead!!
def sentence_bleu(references, hypothesis, max_n=4):
    weights = [1.0 / max_n] * max_n
    precisions = []

    for n in range(1, max_n + 1):
        p_n = modified_precision(references, hypothesis, n)
        if p_n == 0:
            return 0.0  # standard BLEU has no smoothing by default
        precisions.append(math.log(p_n))

    bleu = math.exp(sum(w * p for w, p in zip(weights, precisions)))
    bleu *= brevity_penalty(references, hypothesis)
    return bleu

# def sentence_bleu(references, hypothesis, max_n=4):
#     weights = [1.0 / max_n] * max_n
#     bleu_score = 0.0
#     for n in range(1, max_n + 1):
#         bleu_score += weights[n - 1] * modified_precision(references, hypothesis, n)
#     bleu_score *= brevity_penalty(references, hypothesis)
#     return bleu_score


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
    ref_lens = [len(ref) for ref in references]
    hyp_len = len(hypothesis)
    closest_ref_len = min(ref_lens, key=lambda ref_len: (abs(ref_len - hyp_len), ref_len))
    if hyp_len == 0:
        return 0.0
    if hyp_len > closest_ref_len:
        return 1.0
    else:
        return math.exp(1 - closest_ref_len / hyp_len)

# Paul implemented the brevity function wrongly...
# def brevity_penalty(references, hypothesis):
#     closest_ref_length = min(abs(len(hypothesis) - len(ref)) for ref in references)
#     if closest_ref_length > 0:
#         return math.exp(1 - len(references[0]) / len(hypothesis))
#     else:
#         return 1.0


def ngrams(sentence, n):
    return [tuple(sentence[i:i + n]) for i in range(len(sentence) - n + 1)]


if __name__ == '__main__':
    hypothesis = [['Hello', 'I', 'am', 'Groot']]
    reference = [['Hello', 'I', 'am', 'Johannes']]

    print(bleu_score(hypothesis, reference, max_n = 4))
    print(compute_bleu_score(hypothesis, reference))

    hypothesis = [[1, 2, 3, 4, 5, 6]]
    reference = [[1, 2, 3, 5, 6]]

    # print(bleu_score(hypothesis, reference, max_n = 4))
    print(compute_bleu_score(hypothesis, reference))
    print(-1)