from Features import Features

def load_train_data_and_create_features(path):
    with open(path) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    words_set, tags_set = get_words_and_tags_set(content)
    features = Features(words_set,tags_set)

    word_possible_labels = {}
    words = []
    tags = []
    features = []
    for idx, line in enumerate(content):
        #uncomment if we want to add ** in start of santance
        #words.extend(['*', '*'])
        #tags.extend(['*', '*'])
        for word_tag in line.split():
            word, tag = word_tag.split('_')
            words.append(word)
            tags.append(tag)
            #if word exists append tag to wotds list, else create a list and append the tag
            word_possible_labels.setdefault(word, []).extend(tag)
        words.append('STOP')
        tags.append('STOP')

def get_words_and_tags_set(content):
    words_set = set()
    tags_set = set()
    for line in content:
        for word_tag in line.split():
            word, tag = word_tag.split('_')
            words_set.add(word)
            tags_set.add(tag)
    return words_set, tags_set

def create_word_tag_pairs(words_set,tags_set):
    word_tag_pairs = set()
    for tag in tags_set:
        for word in words_set:
            word_tag_pairs.add((tag,word))
    word_tag_pairs.add(('STOP','STOP'))










if __name__ == '__main__':
    load_train_data_and_create_features('data/train.wtag')


