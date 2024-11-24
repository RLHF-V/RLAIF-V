import jsonlines
from nltk.tokenize import word_tokenize

from .filter import Filter


class LengthFilter(Filter):
    """
    Adjust the average length of chosen and rejected to make them similar
    """

    @classmethod
    def count_words(cls, sentence):
        words = word_tokenize(sentence)
        return len(words)

    @classmethod
    def calculate_mean_difference(cls, data):
        total_difference = sum(item['chosen_diff'] for item in data)
        return total_difference / len(data)

    @classmethod
    def do_filter(cls, data):
        for item in data:
            chosen_count = cls.count_words(item['chosen'])
            reject_count = cls.count_words(item['rejected'])
            item['chosen_diff'] = chosen_count - reject_count

        data.sort(key=lambda x: x['chosen_diff'])

        print("finish sorting")
        print("mean difference: ", cls.calculate_mean_difference(data))

        while cls.calculate_mean_difference(data) > 0.5:
            print("pop data to reduce mean difference")
            data.pop()
        for item in data:
            del item['chosen_diff']

        return data
