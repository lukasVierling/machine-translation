from tqdm import tqdm
import pickle as pkl
from datetime import datetime
from multiprocessing import Pool

class BytePairEncoder():   

    #['token token',...] 'toke toke' bedeutet 'token token' -> 'tokentoken'
    __encoding_rules = []
    #start of sentence (sos) and end of sentence token (eos)
    __SOS_TOKEN = '<s>'
    __EOS_TOKEN = '</s>'
    __vocab_size = 0

    @staticmethod
    def load_model(model_path):
        """
        Loads a pre-trained Byte Pair Encoder model from a file.

        Args:
            model_path (str): The path to the pickle file containing the pre-trained model.

        Returns:
            BytePairEncoder: The pre-trained Byte Pair Encoder model.
        """
        with open(model_path, 'rb') as f:
            encoder = pkl.load(f)
        return encoder
    
    def get_vocab_size(self):
        return self.__vocab_size

    def set_rules(self, rules):
        """
        Function for testing BPE encoding.

        Args:
            rules (list): List of rules, that should be used for encoding
        """
        self.__encoding_rules = rules
    
    def save_model(self, file_name = f'../logs/BPE_{datetime.now().strftime("%d-%m_%H:%M:%S")}'):
        """
        Save the current state of the BytePairEncoder object as a pickle file.

        Args:
        - file_name (str): (Optional) The name of the file to save the encoder object to. If not provided, the default file name is used, which includes the current date and time.

        Returns:
        - str: The file name of the saved pickle file.
        """
        with open(f'{file_name}.pickle', 'wb') as f:
            pkl.dump(self, f)
        return f'{file_name}.pickle'
    
    def fit(self, text, operations=5000):
        '''
        Fit the BPE model on the given text data.

        Args:
            text (str) : The input text data to fit the BPE model on.
            operations (int, optional): The maximum number of BPE operations to perform during model fitting, by default 5000.
        
        Returns:
            set(str) : The resulting vocabulary after encoding the input text with the learned BPE
        '''
        self.__encoding_rules = []
        ##Turn text into a list of tokens, seperate sentences with eos token and signal in word relationships with @@
        processed_text = self.__preprocess_text(text)
        # processed_text = List[List[str]] : A list of sentences, each sentence is a list of tokens.
        # Iterate over the specified number of BPE operations.
        frequencies = self.__get_frequencies(processed_text)
        # Create a dictionary to store the frequencies of each word in the text
            # and initialize it with the frequencies of the tokens in the vocabulary.
        progress_bar = tqdm(range(operations), desc="Processing Text")
        for i in progress_bar:
            # Take the input frequency dictionary and create a new dictionary based on it
            # This dictionary has an entry for every pair of adjacent tokens in the keys of the frequencies dictionary
            # A pair of token are two characters seperated by @@.
            pair_frequencies = self.__get_pair_frequencies(frequencies)
            # Get the most frequent pair of tokens in the text
            if not pair_frequencies:
                # Break out of the loop if pair_frequencies is an empty dictionary
                break
            new_pair = max(pair_frequencies, key=pair_frequencies.get)
            # Now merge all occurences of this pair in the keys of the frequencies dictionary and return the modified dictionary
            frequencies = self.__merge_pair(frequencies, new_pair)
            # If the new pair is not already in the vocabulary, add it
            self.__encoding_rules += [new_pair]
            # Add postfix to progress_bar in the form New Rule: new pair
            progress_bar.set_postfix_str(f"New Rule: {new_pair}")
        vocabulary = [word for word in frequencies]
        new_vocab = []
        for word in vocabulary:
            if "@@" in word:
                tokens = word.split("@@")
                word = [token+"@@" for token in tokens[:-1]] + [tokens[-1]]
            else:
                word = [word]
            new_vocab += word
        self.__vocab_size = len(set(new_vocab))
        return set(new_vocab)
                

    
    def encode_corpus(self, corpus):
        """
        Encodes the given corpus using the defined encoding rules.
        
        This method takes a corpus as input, preprocesses it, and encodes each word in the corpus based on the
        defined encoding rules. The encoding process involves replacing each word with its corresponding encoded form
        according to the mapping provided by the encoding rules.
        
        Args:
            corpus (Str): The corpus to be encoded, represented as a string.
        Returns:
            List[List[str]]: The encoded corpus, where each word is replaced with its encoded form.
        """
        corpus = self.__preprocess_text(corpus)

        # create a dictionary for every word in the corpus of the form word:word
        # The dictionary is used to map the words in the corpus to their encoded form
        mapping = {word:word for sentence in corpus for word in sentence}

        """
        Old version:
        mapping = {}
        for sentence in corpus:
            for word in sentence:
                mapping[word] = word
        """
        
        # Encode every word in the corpus
        mapping = {word: self.__encode(word) for word in mapping}
        """
        Old version:
        for word in mapping:
            mapping[word] = self.__encode(word)
        """

        # Replace every word in the corpus with the encoded word in the dictionary
        for i, sentence in enumerate(corpus):
            corpus[i] = [mapping[word] for word in sentence]
        # Transform the corpus into the correct representation of tokens
        new_corpus = []
        for sentence in corpus:
            new_sentence = []
            for word in sentence:
                if "@@" in word:
                    tokens = word.split("@@")
                    word = [token+"@@" for token in tokens[:-1]] + [tokens[-1]]
                else:
                    word = [word]
                
                new_sentence += word
            new_corpus += [new_sentence]

        return new_corpus
    
    def decode_corpus(self,corpus):
        """
        Decodes the given corpus using the defined decoding rules.
        
        This method takes an encoded corpus as input and decodes each word in the corpus based on the defined
        decoding rules. The decoding process involves replacing each encoded word with its original form according
        to the mapping provided by the decoding rules.
        
        Args:
            corpus (List[List[str]]): The encoded corpus to be decoded, represented as a list of sentences where each
                sentence is represented as a list of encoded words.
            
        Returns:
            List[List[str]]: The decoded corpus, where each encoded word is replaced with its original form. The
                decoded corpus has the same structure as the input corpus, i.e., a list of sentences where each sentence
                is a list of decoded words.
        """
        #TODO hier uach wider ein Wort encodiere nund das einsetzen anstatt den ganzen Text zu bearbeiten
        pool = Pool()
        # Decode the corpus
        decoded_corpus = []
        for sentence in corpus:
            decoded_corpus += [self.decode(sentence)]
        return decoded_corpus
    
    def get_rules(self):
        """
        Returns the encoding rules used by the class.
        
        This method retrieves the encoding rules that are stored in the private attribute __encoding_rules. The encoding
        rules define the mapping between characters and their encoded representations.
        
        Returns:
            list: A list of encoding rules.
        """
        return self.__encoding_rules
    
    def decode(self, sentence):
        """
        Decodes a sentence using the Byte Pair Encoder model.

        Args:
            sentence List[str]: The sentence to decode.

        Returns:
            Str: The decoded sentence.
        """
        # Sentence is a list of tokens
        # join the tokens on whitespace and remove all @@ with nothing
        sentence = " ".join(sentence) 
        sentence = sentence.replace("@@ ","")
        # Remove special tokens
        sentence = sentence.replace('<s> ','')
        sentence = sentence.replace('</s>','')
        sentence = sentence[:-1]
        return sentence
    
    def __encode(self, word):
        """
        Encodes a word using the Byte Pair Encoder model.

        Args:
            word Str: The sentence to encode.

        Returns:
            Str: The encoded word.
        """
        # Iterate over the encoding rules in reverse order.
        for rule in self.__encoding_rules:
            word = self.__apply_rule(word, rule)
            if not('@@' in word):
                break
        # Return the encoded sentence.
        return word
    
    def __apply_rule(self, word, rule):
        """
        Applied all encoding rules to a given word.

        Args:
            word (str): The word to apply the rules to, where every character in the word is seperated by @@.
            rule (str): The rule to apply to the word.

        Returns:
            str: The word after applying the rule.
        """
        characters = word.split("@@")
        characters = [character+'@@' for character in characters[:-1]] + [characters[-1]]
        for i in range(len(characters) - 1):
            if characters[i] + characters[i+1] == rule:
                characters[i]  = rule.replace('@@','',1)
                characters[i+1] = ""
        # remove all empty strings from the array
        characters = [character for character in characters if character != ""]
        # merge the array elements into a single string
        new_token = "".join(characters)

        return new_token

    def __merge_pair(self, frequencies, new_pair):
        """
        Private function to merge a pair of tokens in the keys of a dictionary.
        Merging two keys means, that the new key is the concatenation of the two keys and the @@ is removed betwen them

        Args:
            frequencies (Dict[str, int]): The dicitonary with the original frequencies.
            Each key consists of several concataned tokens, with every token separated by @@
            new_pair (str): The pair of tokens to merge.

        Returns:
            Dict[str, int]: The modified dictionary, where each occurence of new_pair is merged
        """
        new_frequencies = {}

        for token in frequencies:
            # Replace all occurences of the new pair in the key with the merged pair
            # Turn String into a list and preseve the @@ 
            characters = token.split("@@")
            characters = [character+'@@' for character in characters[:-1]] + [characters[-1]]
            for i in range(len(characters) - 1):
                if characters[i] + characters[i+1] == new_pair:
                    characters[i] = characters[i].replace('@@','') + characters[i+1]
                    characters[i+1] = ""

            # remove all empty strings from the array
            characters = [character for character in characters if character != ""]
            # merge the array elements into a single string
            new_token = "".join(characters)

            if token in new_frequencies:
                new_frequencies[new_token] += frequencies[token]
            else:
                new_frequencies[new_token] = frequencies[token]

        return new_frequencies
    
    def __get_pair_frequencies(self,frequencies):
        """
        Counts the frequencies of all adjacent character pairs in a given dictionary.

        Args:
            frequencies (Dict[str, int]): The frequencies of the tokens in the corpus.
            The keys are string consisting of several concatenated tokens. every token is seperated by @@

        Returns:
            Dict[str, int]: The frequencies of all adjacent character pairs in the corpus.
        """
        pair_frequencies = {}
        # Count the ocurences for every token pair and add them to the dictionary
        for token in frequencies:
            # Split the token into the individual tokens that it consists of
            # Every token is seperated by @@
            characters = token.split("@@")
            characters = [character+'@@' for character in characters[:-1]] + [characters[-1]]
            # Iterate over the characters in the token
            for i in range(len(characters) - 1):
                # Create a pair of adjacent characters
                pair = characters[i] + characters[i+1]
                # Increase freqency of the pair
                if pair in pair_frequencies:
                    pair_frequencies[pair] += frequencies[token]
                else:
                    pair_frequencies[pair] = frequencies[token]

        return pair_frequencies
        
    def __get_frequencies(self,processed_text):
        """
        Gets the frequencies of all tokens in the text.

        Args:
            processed_text List[List[str]]: The text to get the frequencies of.

        Returns:
            Dict[str, int]: A dictionary containing the frequencies of all tokens in the text.
        """
        # Create a dictionary to store the frequencies of each word in the text
        # and initialize it with the frequencies of the tokens in the vocabulary.
        frequencies = {}
        for sentence in processed_text:
            for token in sentence:
                # Increase frequency of the word
                if token in frequencies:
                    frequencies[token] += 1
                else:
                    frequencies[token] = 1
        return frequencies
    
    def __preprocess_text(self,text):
        '''
        Turn text into a list of tokens, seperate sentences with eos and sos token and signal in word relationships with @@
        
        Args:
            text (str) : The input text data to fit the BPE model on.
        
        Returns:
            List[List[str]] : A list of sentences, each sentence is a list of tokens.
        '''
        #split text into sentences
        sentences = text.split('\n')
        #split sentences into words
        sentences = [sentence.split(' ') for sentence in sentences]
        processed_sentences = []
        for sentence in sentences:
            # Add start-of-sentence token and end-of-sentence token
            processed_words = ["<s>"] + [ '@@'.join(list(word)) for word in sentence ] + ["</s>"]
            processed_sentences.append(processed_words)
        #turn words into a list of tokens
        #signal in word relationships with @@
        return processed_sentences

