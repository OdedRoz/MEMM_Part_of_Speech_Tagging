class Viterbi:
    @staticmethod
    def print(V):
        for y in V[0].keys():
            print('State:', y)
            for t in range(len(V)):
                print(V[t][y])

    @staticmethod
    def viterbi_for_memm(
            features_list,
            word_list,
            states,
            train_probabilities,
            smoothing_factor,
            feature_obj,
            features_on_the_fly = True):
        V = [{}]
        path = {}
        curr_obs = features_list[0]
        init_history_words = ['*', '*']
        for y in states:
            curr_prob = 1.0
            if (True == features_on_the_fly):
                curr_obs = feature_obj.set_features_for_word(
                    words=init_history_words + [word_list[0]],
                    next_word=word_list[1],
                    tags=init_history_words )
                # print (curr_obs )
            for i in range(len(curr_obs)):
                curr_feature = curr_obs[i]
                if curr_feature in train_probabilities[y]:
                    curr_prob = curr_prob * train_probabilities[y][curr_feature]
                else:
                    curr_prob = curr_prob * smoothing_factor
            # for probability not run to zero
            # curr_prob = curr_prob
            V[0][y] = curr_prob
            path[y] = [y]

        print('Viterbi Number Of Obs:' + str(len(features_list)))
        verge_factor = 10 ** (-100)
        adjust_factor = 10 ** 99

        for t in range(1, len(features_list)):
            if t % 2 == 0:
                if all([val < verge_factor for val in V[t - 1].values()]):
                    # for probability not run to zero
                    print('Making Probability Adjust')
                    for y0 in states:
                        V[t - 1][y0] = V[t - 1][y0] * adjust_factor
            V.append({})
            new_path = {}
            curr_obs = features_list[t]
            if t == 1:
                hist_index = 1
            else:
                hist_index = 0
            for y in states:
                max_prob = - 1
                former_state = None
                for y0 in states:
                    curr_prob = V[t - 1][y0]
                    if (True == features_on_the_fly):
                        curr_obs = feature_obj.set_features_for_word(
                            words=['*']*hist_index + word_list[t-(2 - hist_index):t+1],
                            next_word=word_list[t+1],
                            tags=['*']*hist_index + path[y0][(-2 + hist_index):])
                        # print(curr_obs)
                    for i in range(len(curr_obs)):
                        curr_feature = curr_obs[i]
                        if curr_feature in train_probabilities[y]:
                            curr_prob = curr_prob * train_probabilities[y][curr_feature]
                        else:
                            curr_prob = curr_prob * smoothing_factor

                    if curr_prob > max_prob:
                        max_prob = curr_prob
                        former_state = y0
                V[t][y] = max_prob
                new_path[y] = path[former_state] + [y]

            path = new_path

        prob = -1
        for y in states:
            cur_prob = V[len(features_list) - 1][y]
            if cur_prob > prob:
                prob = cur_prob
                state = y

        return prob, path[state]