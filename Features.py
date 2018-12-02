class Features:
    def __init__(self,words_set, tags_set):
        self.words_set = words_set
        self.tags_set = tags_set
        self.f100 = self.create_word_tag_pairs()
        self.f101 = self.create_suffixes()
        self.f102 = self.create_prefixes()
        self.f103 = self.create_three_tags()
        self.f104 = self.create_two_tags()
        self.f105 = tags_set

    def create_word_tag_pairs(self):
        word_tag_pairs = set()
        for tag in self.tags_set:
            for word in self.words_set:
                word_tag_pairs.add((tag, word))
        word_tag_pairs.add(('STOP', 'STOP'))
        word_tag_pairs.add(('*', '*'))
        return word_tag_pairs

    def create_suffixes(self):
        suffixes_sizes = [1,2,3,4]
        suffixes = set()
        for word in self.words_set:
            for size in suffixes_sizes:
                suffixes.add(word[:size])
        return suffixes

    def create_prefixes(self):
        prefixes_sizes = [1,2,3,4]
        prefixes = set()
        for word in self.words_set:
            for size in prefixes_sizes:
                prefixes.add(word[-size:])
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





