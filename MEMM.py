
import os
import time
from math import exp
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from load_corpus import load_data_and_create_features
from Viterbi import Viterbi


###################################
# general note for understanding: #
# state and tag has same meaning  #
###################################

class MEMM():
    def __init__(
            self,
            train_filename = "train.wtag",
            test_filename  = "test.wtag"):

        # init params
        self.number_of_gradient_decent_steps = 3
        self.regularization_factor           = 0.5
        self.learning_rate                   = 0.01
        # load train in requested format
        train_words, train_tags, train_features, feat_obj = load_data_and_create_features(
                os.path.join('data', train_filename))

        test_words, test_tags, test_features, feat_obj = load_data_and_create_features(
                os.path.join('data', test_filename), dataset = 'Test', Features_Object = feat_obj)

        self.train_model(
                train_words,
                train_tags,
                train_features,
                optimize_with_manual_ga = True)


        output_words, output_pred, actual_tags = self.test_dataset(
                test_words=test_words,
                test_features=test_features,
                test_tags=test_tags,
                all_states=self.all_tags,
                probabilities_dict=self.state_feature_probability_dict,
                smoothing_factor=1/float(len(self.all_tags)))
        total = 0.0
        correct = 0.0
        for i in range(len(output_pred)):
            total += 1
            if output_pred[i] == actual_tags[i]:
                correct += 1
        print('Total / correct: [' + str(total)+ ' : ' + str(correct) + ']')
        print("Acuuracy : " + str(correct/total))

    def train_model(
            self,
            train_words,
            train_tags,
            train_features,
            optimize_with_manual_ga = False):

        train_start_time = time.time()
        # init useful dictionaries
        state_features_occurrences = {}
        feature_count_dict = {}
        all_features = []
        all_tags = list(set(train_tags))
        for temp_tag in all_tags:
            state_features_occurrences[temp_tag] = {}

        # iterate all words and create empirical counts
        for i in range(len(train_words)):
            curr_word    = train_words[i] # not needed
            curr_tag     = train_tags[i]
            curr_features= train_features[i]

            if ('STOP' == curr_word) or ('*' == curr_word):
                continue

            all_features.extend(list(curr_features))
            for curr_feat in curr_features:
                # for each tag get the number of times each feature let to it
                if curr_feat not in state_features_occurrences[curr_tag]:
                    state_features_occurrences[curr_tag][curr_feat] = 1
                else:
                    state_features_occurrences[curr_tag][curr_feat] += 1
                # count how many times each feature appeared
                if curr_feat not in feature_count_dict:
                    feature_count_dict[curr_feat] = 1
                else:
                    feature_count_dict[curr_feat] += 1

        # adds to every state zero counts for features that didn't appear with it
        all_features = list(set(all_features))
        for state in state_features_occurrences:
            for feature in all_features:
                if feature not in state_features_occurrences[state]:
                    state_features_occurrences[state][feature] = 0.0
        print('Finished empiric counts...')
        # get wieghts per tag|feature
        if (True == optimize_with_manual_ga):
            state_feature_weight_dict = self.calc_feature_weights_manual_ga(
                    all_tags,
                    feature_count_dict,
                    state_features_occurrences)
        else:
            self.calc_feature_weights_scipy_optim(
                    all_tags,
                    all_features,
                    feature_count_dict,
                    state_features_occurrences)

        print('Finished weight calc...')
        # create probabilities from for all p(state | obs)
        state_feature_probability_dict = {}
        for outer_state in all_tags:
            state_feature_probability_dict[outer_state] = {}
            for obs in state_feature_weight_dict[outer_state]:
                probability_denominator = 0.0
                for inner_state in all_tags:
                    if obs in state_feature_weight_dict[inner_state]:
                        probability_denominator += exp(state_feature_weight_dict[inner_state][obs])
                # creates the probability - might need smoothing
                        state_feature_probability_dict[outer_state][obs] = exp(state_feature_weight_dict[outer_state][obs]) / float(
                    probability_denominator)
        # saves leared probabilities in self argument
        self.state_feature_probability_dict =  state_feature_probability_dict
        print('Finished trainig in ' + str(time.time() - train_start_time) + " Scondes" )

    def calc_feature_weights_manual_ga(
            self,
            all_states,
            feature_count_dict,
            state_feature_count_dict):
        """
        :param all_states: list of all possible states
        :param feature_count_dict: dictionary containing how many times each feature observation occurred in train
        :param state_feature_count_dict: dictionary containing how many times each state has each feature observation
         lead to it in train
        :return: for each state and observation in state_feature_count_dict dict containing learned weight
        """
        self.all_tags = all_states
        state_feature_weight_dict = {}
        # init all weights to 0
        for state in state_feature_count_dict:
            state_feature_weight_dict[state] = {}
            for obs in state_feature_count_dict[state]:
                state_feature_weight_dict[state][obs] = 0.0

        # the loop below preforms gradient accent to find the state-feature weights
        for step in range(self.number_of_gradient_decent_steps):
            for state in state_feature_weight_dict:
                for outer_obs in state_feature_weight_dict[state]:
                    # the last step (state, obs) weight
                    curr_weight = state_feature_weight_dict[state][outer_obs]
                    # the number of times obs lead to state
                    curr_empirical_count = state_feature_count_dict[state][outer_obs]
                    # expected_count from the partial derivative formula
                    expected_count = 0.0
                    denominator = 0.0
                    for inner_state in all_states:
                        if outer_obs in state_feature_count_dict[inner_state]:
                            amount_appeared_together = state_feature_count_dict[inner_state][outer_obs]
                            denominator += amount_appeared_together * exp(curr_weight)
                            denominator += (feature_count_dict[outer_obs] - amount_appeared_together) * exp(0)
                        else:
                            # print('Not Supposed To Get Here')
                            denominator += feature_count_dict[outer_obs] * exp(0)

                    for inner_state in all_states:
                        if outer_obs in state_feature_count_dict[inner_state]:
                            numerator = state_feature_count_dict[inner_state][outer_obs] * exp(curr_weight)
                            expected_count += numerator / float(denominator)
                    # the last part of the formula is to avoid overfitting
                    curr_partial_derivative = float(curr_empirical_count) - expected_count - curr_weight / float(
                        self.regularization_factor)
                    # if curr_weight != 0.0:
                    #     print (str(curr_weight ))
                    state_feature_weight_dict[state][
                        outer_obs] = curr_weight + self.learning_rate * curr_partial_derivative
            print('Gradient Accent Step Finished: [' + str(step + 1) + " : " + str(
                self.number_of_gradient_decent_steps) + "]")
        return state_feature_weight_dict

    def calc_feature_weights_scipy_optim(
            self,
            all_tags,
            all_features,
            feature_count_dict,
            state_features_occurrences):

        # add useful parameters to self object
        self.create_feature_index_to_state_feature_mapping(all_tags, all_features)
        self.state_features_occurrences = state_features_occurrences
        self.feature_count_dict         = feature_count_dict
        self.all_tags                   = all_tags
        # init weights to zero
        self.state_feature_weight_arr = np.zeros(len(self.index_to_state_feature_list), dtype=np.float64)
        optimal_params = fmin_l_bfgs_b(func=self.loss_func_and_gradient, x0=self.state_feature_weight_arr)

        if optimal_params[2]['warnflag']:
            print('Error in training:\n{}\\n'.format(optimal_params[2]['task']))

        res_weights = optimal_params[0]

        state_feature_weight_dict = {}
        # init all weights to 0
        for state in all_tags:
            state_feature_weight_dict[state] = {}
            for obs in all_features:
                state_feature_weight_dict[state][obs] = res_weights[self.state_feature_to_index_dict[state][obs]]

        return state_feature_weight_dict

    def loss_func_and_gradient(
            self,
            weights):

        # init empirical counts
        all_empirical_counts    = np.zeros(len(self.index_to_state_feature_list), dtype=np.float64)
        # init partial deteratives
        all_partial_deteratives = np.zeros(len(self.index_to_state_feature_list), dtype=np.float64)

        expected_loss_feature_dict = {}

        for feature_indx in range(len(self.index_to_state_feature_list)):
            curr_state_feature = self.index_to_state_feature_list[feature_indx]
            curr_state   = curr_state_feature[0]
            curr_feature = curr_state_feature[1]
            curr_weight  = weights[feature_indx]
            curr_empirical_count = self.state_features_occurrences[curr_state][curr_feature]

            if curr_feature not in expected_loss_feature_dict:
                if self.state_features_occurrences[curr_state][curr_feature] > 0:
                    expected_loss_feature_dict[curr_feature] = [
                        self.state_features_occurrences[curr_state][curr_feature]*exp(curr_weight)]
                else:
                    expected_loss_feature_dict[curr_feature] = [exp(0)]
            else:
                if self.state_features_occurrences[curr_state][curr_feature] > 0:
                    expected_loss_feature_dict[curr_feature].append(
                        self.state_features_occurrences[curr_state][curr_feature]*exp(curr_weight))
                else:
                    expected_loss_feature_dict[curr_feature].append(exp(0))

            all_empirical_counts[feature_indx] = curr_empirical_count

            expected_count = 0.0
            denominator = 0.0
            for inner_state in self.all_tags:
                if curr_feature in self.state_features_occurrences[inner_state]:
                    amount_appeared_together = self.state_features_occurrences[inner_state][curr_feature]
                    denominator += amount_appeared_together * exp(curr_weight)
                    denominator += (self.feature_count_dict[curr_feature] - amount_appeared_together) * exp(0)
                else:
                    print('Not Supposed To Get Here')
                    # denominator += feature_count_dict[outer_obs] * exp(0)

            for inner_state in self.all_tags:
                if curr_feature in self.state_features_occurrences[inner_state]:
                    numerator = self.state_features_occurrences[inner_state][curr_feature] * exp(curr_weight)
                    expected_count += numerator / float(denominator)

            # the last part of the formula is to avoid overfitting
            curr_partial_derivative = float(curr_empirical_count) - expected_count - curr_weight / float(
                self.regularization_factor)

            all_partial_deteratives[feature_indx] = curr_partial_derivative

        expected_loss = 0.0

        for feature in expected_loss_feature_dict:
            expected_loss += np.log(sum(expected_loss_feature_dict[feature]))

        regularization_loss = (np.sum(np.square(weights)) * self.regularization_factor / 2)
        loss = np.sum(weights * all_empirical_counts) - expected_loss - regularization_loss

        print('Finished Optimization Step Loss: [' + str((-1)*loss) + "]")

        return ((-1)*loss), (-1)*all_partial_deteratives

    def create_feature_index_to_state_feature_mapping(
            self,
            all_tags,
            all_features):
        """
        :param all_tags: list of all unique tags (states)
        :param all_features: list of all unique features
        :return: creates mapping between state and feature to an feature list index
                 and vice versa
        """
        self.index_to_state_feature_list = []
        self.state_feature_to_index_dict = {}
        i = 0

        for state in all_tags:
            self.state_feature_to_index_dict[state] = {}
            for feature in all_features:
                self.index_to_state_feature_list.append((state, feature))
                self.state_feature_to_index_dict[state][feature] = i
                i += 1

    def test_dataset(
            self,
            test_words,
            test_features,
            all_states,
            probabilities_dict,
            smoothing_factor,
            test_tags = None):

        output_words          = []
        output_pred           = []
        num_of_processed_sent = 0

        if test_tags is not None:
            actual_tags = []
        else:
            actual_tags = None

        temp_words      = []
        temp_featrues   = []
        for i in range(len(test_words)):
            if test_words[i] == 'STOP':
                vt_res = Viterbi.viterbi_for_memm(features_list=tuple(temp_featrues),
                                                  states=tuple(all_states),
                                                  train_probabilities=probabilities_dict,
                                                  smoothing_factor=smoothing_factor)

                output_words.extend(temp_words)
                output_pred.extend(vt_res[1])
                num_of_processed_sent += 1
                print(
                    'Sentence Processed :' + str(num_of_processed_sent) + ' , Viterbi Probabilities: ' + str(vt_res[0]))

                temp_words = []
                temp_featrues = []

            elif test_words[i] == '*':
                continue

            else:
                temp_words.append(test_words[i])
                temp_featrues.append(test_features[i])
                if test_tags is not None:
                    actual_tags.append(test_tags[i])

        return output_words, output_pred, actual_tags


if __name__ == '__main__':
    model = MEMM()