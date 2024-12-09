from .filter import Filter
from nltk.tokenize import word_tokenize
import random


class NumFilter(Filter):
    """
    Control the amount of data corresponding to an image
    """

    @classmethod
    def count_words(cls, sentence):
        words = word_tokenize(sentence)
        return len(words)

    @classmethod
    def do_filter(cls, data):
        count = {}
        sign = 0  # caption数据个数

        num_filter_out = []
        sum_chosen = 0
        sum_rejected = 0
        total_samples = 0
        # shuffle data
        list_data = data
        random.shuffle(list_data)
        for data in list_data:
            if data["chosen"] == data["rejected"]:
                continue
            idx = data["idx"]
            if idx in count:
                if count[idx] >= 3:
                    continue
            else:
                count[idx] = 0
            count[idx] += 1

            chosen_words = cls.count_words(data["chosen"])
            rejected_words = cls.count_words(data["rejected"])

            if chosen_words > 100:
                sign += 1
            # elif chosen_words < 50:
            #     if(random.random() < 0.35):
            #         continue

            sum_chosen += chosen_words
            sum_rejected += rejected_words
            total_samples += 1

            num_filter_out.append(data)

        return num_filter_out
