import numpy as np

class Features:
    def __init__(self,words_set, tags_set):
        self.words_set = words_set
        self.tags_set = tags_set
        self.f100 = self.create_word_tag_pairs()
        self.f100_dict = self.create_features_dict(self.f100)
        self.f101 = self.create_suffixes()
        self.f101_dict = self.create_features_dict(self.f101)
        self.f102 = self.create_prefixes()
        self.f102_dict = self.create_features_dict(self.f102)
        self.f103 = self.create_three_tags()
        self.f103_dict = self.create_features_dict(self.f103)
        self.f104 = self.create_two_tags()
        self.f104_dict = self.create_features_dict(self.f104)
        self.f105 = tags_set
        self.f105_dict = self.create_features_dict(self.f105)
        self.features_size = self.get_features_size()
        self.features_to_weighet_dict = self.create_features_to_weighet_dict()
        self.weights = np.zeros(self.features_size)

    def create_features_to_weighet_dict(self):
        pass

    def get_features_size(self):
        return sum([len(self.f100),
                    len(self.f101),
                    len(self.f102),
                    len(self.f103),
                    len(self.f104),
                    len(self.f105)])

    def set_features_for_word(self,words,tags):
        word = words[-1]
        tag = tags[-1]
        f100 = self.f100_dict[(word,tag)]
        sufpresize = [1,2,3,4]
        f101 = list()
        f102 = list()
        for size in sufpresize:
            if len(words[-1]) >= size:
                f101.append(self.f101_dict[word[-size:]])
                f102.append(self.f102_dict[word[:size]])
        f103 = self.f103_dict[(tags[0],tags[1],tags[2])]
        f104 = self.f104_dict[(tags[1],tags[2])]
        f105 = self.f105_dict[(tags[2])]
        return {'f101': f101,
                'f102': f102,
                'f103': f103,
                'f104': f104,
                'f105': f105}


    def create_features_dict(self,features):
        features_dict = {}
        for idx,feature in enumerate(features):
            features_dict[feature] = idx
        return features_dict



    def create_word_tag_pairs(self):
        word_tag_pairs = set()
        for tag in self.tags_set:
            for word in self.words_set:
                word_tag_pairs.add((word,tag))
        #word_tag_pairs.add(('STOP', 'STOP'))
        #word_tag_pairs.add(('*', '*'))
        return word_tag_pairs

    def create_suffixes(self):
        suffixes_sizes = [1,2,3,4]
        suffixes = set()
        for word in self.words_set:
            for size in suffixes_sizes:
                suffixes.add(word[-size:])
        return suffixes

    def create_prefixes(self):
        prefixes_sizes = [1,2,3,4]
        prefixes = set()
        for word in self.words_set:
            for size in prefixes_sizes:
                prefixes.add(word[:size])
        return prefixes

    def create_three_tags(self):
        three_tags = set()
        for tag1 in self.tags_set:
            for tag2 in self.tags_set:
                for tag3 in self.tags_set:
                    three_tags.add((tag1,tag2,tag3))
        return three_tags

    def create_two_tags(self):
        two_tags = set()
        for tag1 in self.tags_set:
            for tag2 in self.tags_set:
                two_tags.add((tag1,tag2))
        return two_tags





