import sys
sys.path.append('../..')

import argparse

import numpy as np
from math import log, exp, e
from src.data_loader import DataLoader
from src.utils.levenshtein import Levensthein

class Metrics():

     
    def __brevity_penalty(self, hypothesis_tokens, reference_tokens):
        """
        Calculate the Brevity Penalty for a given hypothessi and reference.

        Args:
            hypothesis_tokens ([[String]]): List of hypothesis encoded as tokens
            reference_file ([[String]]): List of references encoded as tokens

        Returns:
            int: Brevity Penalty value.
        """
        #Flatten the whole text and calculate the length [[String]] -> [String]
        r = len([token for sentence in reference_tokens for token in sentence])
        c = len([token for sentence in hypothesis_tokens for token in sentence])
        if (c > r):
            return 1
        else:
            return exp(1-(r/c))
    
    def __n_grams(self, n, tokens):
        """
        Generate list with all n-gram elements in a given file.

        Args:
            n (int): length of n-grams.
            file ([[String]]): Sentence encoded as tokens

        Returns:
        list[int]: list containing all n-grams with length n.
        """
        #join n tokens to one n_gram for all tokens
        n_grams_array = [' '.join(tokens[i:i+n]) for i in range(0,len(tokens)-(n-1))]
        return n_grams_array
    
    def __n_gram_intersection(self, n, hypothesis_tokens, reference_tokens):
        """
        Calculate the total number of n-grams from the hypothesis in the reference.

        Args:
            n (int): length of n-grams.
            hypothesis_file ([String]): Hypothesis encoded as tokens
            reference_file ([String]): Reference encoded as tokens

        Returns:
        int: amount of n-grams that are in the hypothesis and in the reference.
        """
        #Create n_grams by concatenating n tokens
        reference_n_grams = self.__n_grams(n,reference_tokens)
        hypothesis_n_grams = self.__n_grams(n,hypothesis_tokens)
        #remove all duplicates (list -> set -> list)
        hypothesis_n_grams_unique = list(set(hypothesis_n_grams))
        #count occurences of word in hypothesis that occur in the reference
        #e.g. "the" occurs 3 times in Hyp but 2 times in ref. then we are ablte to map the->the, the->the, the->0 so we add 2=min(3,2) to the whole sum
        counter = 0
        for n_gram in hypothesis_n_grams_unique:
            counter += min(hypothesis_n_grams.count(n_gram),reference_n_grams.count(n_gram))
        return counter
    
    def __modified_n_gram_precision(self, n ,files):
        """
        Calculate the modified n-gram precision P_n, where
            P_n = numerator/denominator
            numerator = \sum_{l=1}^L (\sum_{n-gram \in h_l}[min{# n_gram in h_l, # n_gram in r_l}])
            denominator = \sum_{l=1}^L (\sum_{n-gram \in h_l}[# n_gram in h_l])

        Args:
            n (int): length of n-grams.
            files (list[(String,String)]): list containing tuples of reference and hypothesis of the form
                [(r_1, h_1), ..., (r_L, h_L)].

        Returns:
        int: modified n-gram precision score.
        """
        upper = 0
        lower = 0
        #sum over all reference,hypothesis tuples in the file
        for reference_file,hypothesis_file in files:
            upper += self.__n_gram_intersection(n,hypothesis_file,reference_file)
            lower += len(self.__n_grams(n,hypothesis_file))
        if(lower == 0):
            print('Hypothesis has no n_grams (div by 0)')
            return 0
        return upper/lower
    
    def bleu_score(self,n,hypothesis,reference):
        """
        Calculate the BLEU score of the hypothesis file compared to the reference file, where
            BLEU = BP * 1/n * exp(\sum_i=1^n ln(P_i))
        with BP being the brevity penalty and P_i being the modified i-gram precision

        Args:
            n (int): length of n-grams.
            hypothesis_file [[String]]: List containing the hypothesis.
            reference_file [[String]]: List containing the references.

        Returns:
        int: BLEU score.
        """
        # preprocess input by tokenizing each sentence
        reference_file = [sentence.split() for sentence in reference]
        hypothesis_file = [sentence.split() for sentence in hypothesis]

        #Calculate bp
        bp = self.__brevity_penalty(hypothesis_file,reference_file)
        #Create the correct format needed for the modified_n_gram_prec. function
        tuples = [(ref,hyp) for ref,hyp in zip(reference_file,hypothesis_file)]
        cum = 0
        #sum over all n as in the formula
        for i in range(1,n+1):
            p = self.__modified_n_gram_precision(i,tuples)
            # Changed from p<=0 -> Error to ignore p<= 0 without error message
            if (p>0):
                cum += log(p,e)
        #instead of *(1/n) in every summand, we just calculate *(1/n) in the end (DistributivG.)
        cum = cum * (1/n)
        return (bp * exp(cum))

    def WER(self, hypothesis, reference):
        """
        Calculates the WER (Word Error Rate) of the hypothesis compared to the reference, 
        where
            WER = Levensthein_Distance/Reference_Length. 
        Levensthein_Distance and Reference_Length are calculated sentence-wise 
        and then accumulated.

        Args:
            hypothesis (List[str]): List of Strings representing the hypothesis, 
                where each string represents one sentence
            reference (str): List of Strings representing the reference,
                where each string represents one sentence.

        Returns:
            float: The WER of the hypothesis compared to the reference
        """

        lv = Levensthein()
        levensthein_distance = 0
        ref_length = 0
        
        # calculate levensthein distance and reference length for every sentence and accumulate
        for (hyp, ref) in zip(hypothesis, reference):
            levensthein_distance += lv.levensthein_distance(hyp, ref)
            ref_length += len(ref.split())

        # return WER
        return levensthein_distance/ref_length

    def PER(self, hypothesis, reference):
        """
        Calculates the PER (Position-independent Error Rate) of the hypothesis compared to the reference, 
        where 
            PER = (Matches - max(0, hypothesis_length - reference_length))/reference_Length.
        Matches is the number of words in the hypothesis which is also found in reference and
        is calculated sentence-wise and accumulated.

        Args:
            hypothesis (List[str]): List of Strings representing the hypothesis, 
                where each string represents one sentence
            reference (str): List of Strings representing the reference,
                where each string represents one sentence.

        Returns:
            float: The PER of the hypothesis compared to the reference
        """
        matches = 0
        hyp_length = 0
        ref_length = 0

        # calculate matches, hypothesis length and reference length for every sentence and accumulate
        for (hyp, ref) in zip(hypothesis, reference):
            matches += self.__getMatches(hyp, ref)
            hyp_length += len(hyp.split())
            ref_length += len(ref.split())
        
        PER = 1 - (matches - max(0, hyp_length - ref_length))/ref_length
        return PER
        
    def __getMatches(self, hyp, ref):
        """
        Private method to calculate the matches of the hypothesis given reference (amount of words
        in hypothesis that can also be found in reference). This is basically a wrapper for the 
        __n_gram_intersection function with n = 1.

        Args:
            hyp (str): String representing the hypothesis sentence
            ref (str): String representing the reference sentence

        Returns:
            int: The amount of matches
        """
        # transform reference and hypothesis strings into list of words
        ref_list = ref.split()
        hyp_list = hyp.split()
        
        return self.__n_gram_intersection(1, hyp_list, ref_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-H", "--hypothesis")
    parser.add_argument("-R", "--reference")
    args = parser.parse_args()

    hyp_path = args.hypothesis
    ref_path = args.reference
    
    print("Loading data...")
    hyp_data_loader = DataLoader(hyp_path)
    ref_data_loader = DataLoader(ref_path)

    # test if input path's are valid (NOT VERY SECURE)
    # WARNING: NOT VERY SECURE, POSSIBLE SECURITY VULNERABILITY
    try:
        hyp_data = hyp_data_loader.tokenize(mode="lines")
        ref_data = ref_data_loader.tokenize(mode="lines")
    except(FileNotFoundError):
        print("Error: One or both of the files given are no valid input paths.")
        print("Exiting...")
        exit()
    
    print("Loading data was successful!\n\nCalculating WER...")
    metrics = Metrics()
    WER = metrics.WER(hyp_data, ref_data)
    print(f"WER: {WER}\n")

    print("Calculating PER...")
    PER = metrics.PER(hyp_data, ref_data)
    print(f"PER: {PER}\n")

    print("Calculating BLEU...")
    BLEU = metrics.bleu_score(4, hyp_data, ref_data)
    print(f"BLEU: {BLEU}")

    
    