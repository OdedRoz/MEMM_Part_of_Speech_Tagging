import numpy as np

class Features2:
    def __init__(self,words_set, tags_set):
        self.words_set = words_set
        self.tags_set = tags_set
        self.all_word_tag_pairs = self.create_word_tag_pairs()
        self.f100 = self.words_set
        self.f100_dict = self.create_features_dict(self.f100)
        self.f101 = self.create_suffixes()
        self.f101_dict = self.create_features_dict(self.f101)
        self.f102 = self.create_prefixes()
        self.f102_dict = self.create_features_dict(self.f102)
        self.f103 = self.create_two_tags()
        self.f103_dict = self.create_features_dict(self.f103)
        self.f104 = self.tags_set
        self.f104_dict = self.create_features_dict(self.f104)
        self.f105 = ['current_word_tag']
        self.f105_dict = self.create_features_dict(self.f105)
        #previous word
        self.f106 = self.words_set
        self.f106_dict = self.create_features_dict(self.f106)
        #next word
        self.f107 = self.words_set
        self.f107_dict = self.create_features_dict(self.f107)
        # word form: X for capitalized Xx for mixed d for digit and etc
        self.f_capital = self.craete_word_forms()
        self.f_capital_dict = self.create_features_dict(self.f_capital)
        self.f_pre_w_and_tag = self.create_word_tag_set()
        self.f_pre_w_and_tag_dict = self.create_features_dict(self.f_pre_w_and_tag)
        self.all_features_dicts = [self.f100_dict,
                                   self.f101_dict,
                                   self.f102_dict,
                                   self.f103_dict,
                                   self.f104_dict,
                                   self.f105_dict,
                                   self.f106_dict,
                                   self.f107_dict,
                                   self.f_capital_dict,
                                   self.f_pre_w_and_tag_dict]
        self.features_dicts_sizes = self.get_features_dicts_sizes()
        print(self.features_dicts_sizes)
        self.features_size = self.get_features_size()
        # self.weights = np.zeros(self.features_size)
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
                'f107': len(self.f107_dict),
                'f_capital' : len(self.f_capital_dict),
                'f_pre_w_and_tag' :len(self.f_pre_w_and_tag_dict)}


    def features_to_weighets_index(self,word_dicts_of_features):
        index_of_weighets = []
        feature_counter = 0
        for fx, features in word_dicts_of_features.items():
            index_of_weighets.extend([(x+feature_counter) for x in features])
            feature_counter += self.features_dicts_sizes[fx]
        #assert feature_counter == self.features_size
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

    def set_features_for_word(self,words,tags,next_word,print_OOV=False):
        """
        return which features are 1 for current words and tags
        :param words: list of 3 words [i-2,i-1,i]
        :param tags: list of 2 tags [i-2,i-1]
        :param next_word: word i+1
        :return: indexes of the feature vectors which are 1
        """
        word = words[-1]
        try:
            f100 = [self.f100_dict[word]]
        except:
            if print_OOV:
                print('({word}) OOV for f100')
            f100 = list()
        sufpresize = [1,2,3,4]
        f101 = list()
        f102 = list()
        for size in sufpresize:
            if len(words[-1]) >= size:
                try:
                    f101.append(self.f101_dict[word[-size:]])
                except:
                    if print_OOV:
                        print('({word[-size:]} OOV for f101')
                try:
                    f102.append(self.f102_dict[word[:size]])
                except:
                    if print_OOV:
                        print('({word[:size]}) OOV for f102')
        try:
            f103 = [self.f103_dict[(tags[0],tags[1])]]
        except:
            if print_OOV:
                print('{tags[0]},{tags[1]} OOV for f103')
            f103 = list()
        try:
            f104 = [self.f104_dict[tags[1]]]
        except:
            if print_OOV:
                print('{tags[1]} OOV for f104')
            f104 = list()
        try:
            f105 = [self.f105_dict['current_word_tag']]
        except:
            if print_OOV:
                print('current_word_tag OOV for f105')
            f105 = list()
        try:
            f106 = [self.f106_dict[words[1]]]
        except:
            if print_OOV:
                print('{words[1]} OOV for f106')
            f106 = list()
        try:
            f107 = [self.f107_dict[next_word]]
        except:
            if print_OOV:
                print('({next_word}) OOV for f107')
            f107 = list()
        try:
            if words[-2] == '*' or words[-2] == '.':
                f_capitalized = [self.f_capital_dict[self.get_word_form(word, is_first=True)]]
            else:
                f_capitalized = [self.f_capital_dict[self.get_word_form(word, is_first=False)]]
        except:
            f_capitalized = list()
        try:
            prev_word_and_tag = list()
            # prev_word_and_tag = [self.f_pre_w_and_tag_dict[(word[-1],tags[1])]]
        except:
            prev_word_and_tag = list()

        dict_of_features =  {'f100': f100,
                             'f101': f101,
                             'f102': f102,
                             'f103': f103,
                             'f104': f104,
                             'f105': f105,
                             'f106': f106,
                             'f107': f107,
                             'f_capital' : f_capitalized,
                             'f_pre_w_and_tag' : prev_word_and_tag}
        features = self.features_to_weighets_index(dict_of_features)
        return features


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


    def craete_word_forms(self):
        word_forms = set()
        for word in self.words_set:
            # add every word form one time as first in the sentance and one time not
            temp_word_form = self.get_word_form(word, is_first=True)
            word_forms.add(temp_word_form)
            temp_word_form = self.get_word_form(word, is_first=False)
            word_forms.add(temp_word_form)
        return word_forms

    def get_word_form(self, word, is_first):
        temp_word_form = ''
        temp_word_len = 0
        for char in word:
            if char.isupper():
                if temp_word_form[-1:] != 'X':
                    temp_word_form += 'X'
            elif char.islower():
                if temp_word_form[-1:] != 'x':
                    temp_word_form += 'x'
            elif char.isdigit():
                if temp_word_form[-1:] != 'd':
                    temp_word_form += 'd'
            else:
                if temp_word_form[-1:] != char:
                    temp_word_form += char
            temp_word_len += 1
        return (temp_word_form, is_first)

    def create_word_tag_set(self):
        word_tag_list = []
        for word in self.words_set:
            for tag in self.tags_set:
                word_tag_list.append((word[-1], tag))
        return set(word_tag_list)