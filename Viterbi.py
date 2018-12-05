class Viterbi:
    @staticmethod
    def print(V):
        for y in V[0].keys():
            print('State:', y)
            for t in range(len(V)):
                print(V[t][y])

    @staticmethod
    def viterbi_for_memm(features_list, states, train_probabilities, smoothing_factor):
        V = [{}]
        path = {}
        curr_obs = features_list[0]

        for y in states:
            curr_prob = 1.0
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
            if t % 20 == 0:
                if all([val < verge_factor for val in V[t - 1].values()]):
                    # for probability not run to zero
                    print('Making Probability Adjust')
                    for y0 in states:
                        V[t - 1][y0] = V[t - 1][y0] * adjust_factor
            V.append({})
            new_path = {}
            curr_obs = features_list[t]
            for y in states:
                max_prob = - 1
                former_state = None
                for y0 in states:
                    curr_prob = V[t - 1][y0]
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