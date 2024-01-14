from sklearn.model_selection import train_test_split


class DatasetSplits(object):
    def __init__(self):
        self.__init__()

    @staticmethod
    def basic_split(dirs, train_size):
        return train_test_split(dirs, train_size=train_size, random_state=42)
