from Features import Features

def load_data_and_create_features(path, dataset='Train', Features_Object = None):
    with open(path) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    words_set, tags_set = get_words_and_tags_set(content)
    if dataset=='Train':
        features_object = Features(words_set,tags_set)
    elif dataset=='Test':
        features_object = Features_Object
    word_possible_labels = {}
    words = []
    tags = []
    features = []
    print('creating words,tags and features')
    for idx, line in enumerate(content):
        #uncomment if we want to add ** in start of santance
        words.extend(['*', '*'])
        tags.extend(['*', '*'])
        features.extend([[],[]])
        splited_line = line.split()
        for i,word_tag in enumerate(splited_line):
            word, tag = word_tag.split('_')
            words.append(word)
            tags.append(tag)
            try:
                next_word, _ = splited_line[i+1].split('_')
            except IndexError:
                next_word = 'STOP'
            current_word_features = features_object.set_features_for_word(words[-3:],tags[-3:],next_word)
            features.append(current_word_features)
            #test
            #features_object.multiply_features_with_weighets(features[-1])
            #if word exists append tag to wotds list, else create a list and append the tag
            #word_possible_labels.setdefault(word, []).append(tag)
        words.append('STOP')
        tags.append('STOP')
        features.extend([[]])
    return words,tags,features,features_object

def get_words_and_tags_set(content):
    words_set = set()
    tags_set = set()
    for line in content:
        for word_tag in line.split():
            word, tag = word_tag.split('_')
            words_set.add(word)
            tags_set.add(tag)
    words_set.add('STOP')
    words_set.add('*')
    tags_set.add('STOP')
    tags_set.add('*')
    return words_set, tags_set

def create_word_tag_pairs(words_set,tags_set):
    word_tag_pairs = set()
    for tag in tags_set:
        for word in words_set:
            word_tag_pairs.add((tag,word))
    word_tag_pairs.add(('STOP','STOP'))



if __name__ == '__main__':
    words, tags, features, Features_object = load_data_and_create_features('data/train.wtag','Train')
    load_data_and_create_features('data/test.wtag', 'Test', Features_object)


