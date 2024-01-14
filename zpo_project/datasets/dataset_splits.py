from sklearn.model_selection import train_test_split


class DatasetSplits(object):
    def __init__(self):
        self.__init__()

    @staticmethod
    def basic_split(dirs):
        return train_test_split(dirs, test_size=0.25, random_state=42)
