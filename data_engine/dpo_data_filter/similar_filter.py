import tqdm

from .filter import Filter
import jieba


def get_ngrams(text, n=10):
    words = list(jieba.cut(text))
    return set([' '.join(words[i:i + n]) for i in range(len(words) - n + 1)])


def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0


def deduplicate_data(data_list, threshold=0.9):
    if 'chosen' in data_list[0] and 'rejected' in data_list[0]:
        chosen_seen = []
        rejected_seen = []
        unique_data = []
        for data in tqdm.tqdm(data_list):
            chosen_ngrams = get_ngrams(data['chosen'])
            rejected_ngrams = get_ngrams(data['rejected'])

            chosen_duplicate = any(jaccard_similarity(chosen_ngrams, seen) > threshold for seen in chosen_seen)
            rejected_duplicate = any(jaccard_similarity(rejected_ngrams, seen) > threshold for seen in rejected_seen)

            if not chosen_duplicate and not rejected_duplicate:
                unique_data.append(data)
                chosen_seen.append(chosen_ngrams)
                rejected_seen.append(rejected_ngrams)

        return unique_data
    else:
        text_seen = []
        unique_data = []
        for data in data_list:
            text_ngrams = get_ngrams(data['text'])

            text_duplicate = any(jaccard_similarity(text_ngrams, seen) > threshold for seen in text_seen)

            if not text_duplicate:
                unique_data.append(data)
                text_seen.append(text_ngrams)

        # print(f"before {len(data_list)}, after {len(unique_data)}")
        return unique_data


class SimilarFilter(Filter):
    @classmethod
    def do_filter(cls, data: list) -> list:
        return deduplicate_data(data)