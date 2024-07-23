import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from scipy import stats as st


class StimulusGenerator:

    def __init__(self, words_df, sentences_df,
                 num_blocks=4,
                 candidate_blocks=[[0], [1], [2], [3], [0, 1], [2, 3], [0, 1, 2, 3]],
                 candidate_block_pairs=[([0], [2]), ([0, 1], [2, 3])],

                 # t-test specifications
                 target_positive_tests=[
                     ("ortho_n_dens_s", "COND", "O+P+", "O-P+"),
                     ("ortho_n_dens_s", "COND", "O+P-", "O-P-"),
                     ("ortho_n_dens_s", "COND", "O-P-", "O+P+"),
                     ("phono_n_dens_s", "COND", "O+P+", "O+P-"),
                     ("phono_n_dens_s", "COND", "O-P+", "O-P-"),
                     ("phono_n_dens_s", "COND", "O-P-", "O+P+"),
                     ("surprisal", "label", "high", "low")],
                 target_null_tests=[
                     ("ortho_n_dens_s", "COND", "O+P+", "O+P-"),
                     ("ortho_n_dens_s", "COND", "O-P+", "O-P-"),
                     ("phono_n_dens_s", "COND", "O+P-", "O-P-"),
                     ("phono_n_dens_s", "COND", "O+P+", "O-P+"),],
                

                 # F-test specifications
                 max_variables: list[str] = [
                     "surprisal", "ortho_n_dens_s", "phono_n_dens_s"],
                 min_variables: list[str] = [
                     "word_length", "n_phon", "n_syll", "sum_bigram", "sum_biphone",
                     "lgsubtlwf", "concrete_m", "aff_val_m", "aff_arou_m",
                     "sentence_length", "entropy"],
                 target_match_between_tests=[
                     ("ortho_n_dens_s", "COND", ("O+P+", "O+P-")),
                     ("ortho_n_dens_s", "COND", ("O-P+", "O-P-")),
                     ("phono_n_dens_s", "COND", ("O+P-", "O-P-")),
                     ("phono_n_dens_s", "COND", ("O+P+", "O-P+"))],

                 # track stats on these variables even though they don't directly affect
                 # the objective function
                 monitor_variables: list[str] = [
                     "ortho_n_freq_s_m", "phono_n_freq_s_m",
                     "ortho_upoint", "phono_upoint",
                     "old20_m", "pld20_m"]):
        self.words = words_df
        self.sentences = sentences_df

        self.num_blocks = num_blocks
        self.candidate_blocks = candidate_blocks
        self.candidate_block_pairs = candidate_block_pairs

        self.target_positive_tests = target_positive_tests
        self.target_null_tests = target_null_tests
        self.target_match_between_tests = target_match_between_tests

        self.max_variables = max_variables
        self.min_variables = min_variables
        self.monitor_variables = monitor_variables
        
    def melt_sentences(self, sentences):
        """
        Prepare a long dataframe describing the given sampled sentences in terms of the
        variables we care about.
        """
        return sentences.reset_index().melt(
            id_vars=["block", "target_word", "label", "COND", "our_cond", "sentence"],
            value_vars=self.max_variables + self.min_variables + self.monitor_variables)

    def get_sample(self, word_sample_size=16):
        """
        Sample a stimulus set.

        Args
        ====
        word_sample_size : int
            Number of words per condition (OND * PND * surprisal) to sample.
        """
        s_words = self.words.groupby("COND").sample(word_sample_size, replace=False)
        # sample 1 context sentence per word+context condition
        s_sentences = pd.merge(self.sentences, s_words, left_on="target_word", right_on="word_us").groupby(["target_word", "label"]).sample(1)
        s_sentences["our_cond"] = s_sentences.COND + " " + s_sentences.label

        cv = StratifiedKFold(self.num_blocks, shuffle=True).split(s_sentences, s_sentences.our_cond)
        s_sentences = pd.concat([s_sentences.iloc[idxs].drop(columns=["block"]) for _, idxs in cv],
                                keys=range(self.num_blocks), names=["block"])

        s_melted = self.melt_sentences(s_sentences)

        return s_words, s_sentences, s_melted

    def get_stats(self, melted_df):
        """
        Compute F-tests across a set of stimulus variables.
        """
        ret = []
        for block_union in self.candidate_blocks:
            study_df = melted_df.loc[melted_df.block.isin(block_union)]
            ret.append(study_df.groupby("variable").apply(lambda x:
                pd.Series(st.f_oneway(*[x.loc[x.our_cond == cond, "value"] for cond in x.our_cond.unique()]),
                        index=["F", "p"]), include_groups=False))
        return pd.concat(ret, keys=range(len(self.candidate_blocks)), names=["block_union"]) \
            .reorder_levels(["variable", "block_union"])

    def objective(self, sentence_df, melted_df, alpha=0.01, verbose=False):
        assert sentence_df.index.names[0] == "block"

        # # each word should appear at most once in each block
        # if sentence_df.groupby(["block", "target_word"]).value_counts().max() > 1:
        #     return -np.inf, "repeated_word"

        ############ dataset-wide tests
        # First ensure that the relevant condition t-tests pass across the whole set
        # we expect differences in the relevant target variables between these conds
        for test_spec in self.target_positive_tests:
            var, grouping_variable, group1, group2 = test_spec
            sentence_df_ = sentence_df.set_index(grouping_variable)
            ttest_t, ttest_p = st.ttest_ind(sentence_df_.loc[group1, var], sentence_df_.loc[group2, var])
            if ttest_p > alpha:
                if verbose:
                    print(f"failed positive t-test for {var} between {group1} and {group2}")
                return -np.inf, ("positive_t", test_spec)
                
        # we expect no significant differences in these variables between these conds
        for test_spec in self.target_null_tests:
            var, grouping_variable, group1, group2 = test_spec
            sentence_df_ = sentence_df.set_index(grouping_variable)
            ttest_t, ttest_p = st.ttest_ind(sentence_df_.loc[group1, var], sentence_df_.loc[group2, var])
            if ttest_p < alpha:
                if verbose:
                    print(f"failed null t-test for {var} between {group1} and {group2}")
                return -np.inf, ("null_t", test_spec)
            

        ######### block-level tests
        # we expect no significant differences in these variables within these cond levels between-block
        
        # first ensure that the previous dataset-wide tests also pass on all possible incremental
        # data collection outcomes
        for block_union in self.candidate_blocks:
            study_df = sentence_df.loc[block_union]
            for test_spec in self.target_positive_tests:
                var, grouping_variable, group1, group2 = test_spec
                study_df_ = study_df.set_index(grouping_variable)
                ttest_t, ttest_p = st.ttest_ind(study_df_.loc[group1, var], study_df_.loc[group2, var])
                if ttest_p > alpha:
                    if verbose:
                        print(f"Block union {block_union} failed t-test for {var} between {group1} and {group2}")
                    return -np.inf, ("block_positive_t", test_spec)

            for test_spec in self.target_null_tests:
                var, grouping_variable, group1, group2 = test_spec
                study_df_ = study_df.set_index(grouping_variable)
                ttest_t, ttest_p = st.ttest_ind(study_df_.loc[group1, var], study_df_.loc[group2, var])
                if ttest_p < alpha:
                    if verbose:
                        print(f"Block union {block_union} failed t-test for {var} between {group1} and {group2}")
                    return -np.inf, ("block_null_t", test_spec)
                
        # now ensure that concepts like "O+" and "P-" mean the same thing across
        # two testing modalities (blocks 0, 1 listening; 2, 3 reading)
        for union1, union2 in self.candidate_block_pairs:
            study_df1 = sentence_df.loc[union1]
            study_df2 = sentence_df.loc[union2]
            for test_spec in self.target_match_between_tests:
                var, grouping_variable, groups = test_spec
                study_df1_ = study_df1.set_index(grouping_variable)
                study_df2_ = study_df2.set_index(grouping_variable)
                ftest_F, ftest_p = st.f_oneway(*([study_df1_.loc[group, var] for group in groups] +
                                                [study_df2_.loc[group, var] for group in groups]))
                if ftest_p < alpha:
                    if verbose:
                        print(f"Block union {union1} vs {union2} failed F-test for {var} between {groups}")
                    return -np.inf, ("block_between_f", test_spec)

        ######### F test
        # Constraints above all passed; just rank the stimulus set by F value
        results = self.get_stats(melted_df)

        # max_p = results.loc[max_variables, "p"].min()
        min_p = results.loc[self.min_variables, "p"].min()

        # it's easy to maximize the max variables -- let's focus on the min variables
        # return max_diff - min_diff
        
        return min_p, None