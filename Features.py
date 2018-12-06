import numpy as np

class Features:
    def __init__(self,words_set, tags_set):
        self.words_set = words_set
        self.tags_set = tags_set
        self.all_word_tad_pairs = self.create_word_tag_pairs()
        self.f100 = self.all_word_tad_pairs
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
        self.f106 = self.all_word_tad_pairs
        self.f106_dict = self.create_features_dict(self.f106)
        self.f107 = self.all_word_tad_pairs
        self.f107_dict = self.create_features_dict(self.f107)
        self.all_features_dicts = [self.f100_dict,
                                   self.f101_dict,
                                   self.f102_dict,
                                   self.f103_dict,
                                   self.f104_dict,
                                   self.f105_dict,
                                   self.f106_dict,
                                   self.f107_dict,]
        self.features_dicts_sizes = self.get_features_dicts_sizes()
        self.features_size = self.get_features_size()
        self.weights = np.zeros(self.features_size)
        #self.features_to_weighet_dict = self.create_features_to_weighet_dict()

    def multiply_features_with_weighets(self,features):
        sum = 0
        for feature in features:
            sum += self.weights[feature]




    def get_features_dicts_sizes(self):
        return {'f100': len(self.f100_dict),
                'f101': len(self.f101_dict),
                'f102': len(self.f102_dict),
                'f103': len(self.f103_dict),
                'f104': len(self.f104_dict),
                'f105': len(self.f105_dict),
                'f106': len(self.f106_dict),
                'f107': len(self.f107_dict)}


    def features_to_weighets_index(self,word_dicts_of_features):
        index_of_weighets = []
        feature_counter = 0
        for fx, features in word_dicts_of_features.items():
            index_of_weighets.extend([(x+feature_counter) for x in features])
            feature_counter += self.features_dicts_sizes[fx]
        return tuple(index_of_weighets)




    def get_features_size(self):
        return sum([len(self.f100),
                    len(self.f101),
                    len(self.f102),
                    len(self.f103),
                    len(self.f104),
                    len(self.f105),
                    len(self.f106),
                    len(self.f107)])

    def set_features_for_word(self,words,tags,next_word):
        word = words[-1]
        tag = tags[-1]
        try:
            f100 = [self.f100_dict[(word,tag)]]
        except:
            print(f'({word},{tag}) OOV for f100')
            f100 = list()
        sufpresize = [1,2,3,4]
        f101 = list()
        f102 = list()
        for size in sufpresize:
            if len(words[-1]) >= size:
                try:
                    f101.append(self.f101_dict[word[-size:]])
                except:
                    print(f'{word[-size:]} OOV for f101')
                try:
                    f102.append(self.f102_dict[word[:size]])
                except:
                    print(f'{word[:size]} OOV for f102')
        try:
            f103 = [self.f103_dict[(tags[0],tags[1],tags[2])]]
        except:
            print(f'{tags[0]},{tags[1]},{tags[2]} OOV for f103')
            f103 = list()
        try:
            f104 = [self.f104_dict[(tags[1],tags[2])]]
        except:
            print(f'{tags[1]},{tags[2]} OOV for f104')
            f104 = list()
        try:
            f105 = [self.f105_dict[(tags[2])]]
        except:
            print(f'{tags[2]} OOV for f105')
            f105 = list()
        try:
            f106 = [self.f106_dict[(words[1],tags[2])]]
        except:
            print(f'({words[1]},{tags[2]}) OOV for f106')
            f106 = list()
        try:
            f107 = [self.f107_dict[(next_word,tags[2])]]
        except:
            print(f'({next_word},{tags[2]}) OOV for f107')
            f107 = list()

        return {'f100': f100,
                'f101': f101,
                'f102': f102,
                'f103': f103,
                'f104': f104,
                'f105': f105,
                'f106': f106,
                'f107': f107}


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





