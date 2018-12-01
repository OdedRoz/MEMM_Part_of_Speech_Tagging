def load_train_data_and_create_features(path):
    with open(path) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    word_possible_labels = {}
    for line in content:
        for word_tag in line.split():
            word, tag = word_tag.split('_')
            #if word exists append tag to wotds list, else create a list and append the tag
            word_possible_labels.setdefault(word, []).append(tag)







if __name__ == '__main__':
    load_train_data_and_create_features('data/train.wtag')


