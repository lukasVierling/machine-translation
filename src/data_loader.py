import gzip

class DataLoader():

    def __init__(self,data_file):
        self.data_file = data_file

    def load_data(self):
        '''
        Load the data

        Args:

        Returns:
            String: the text in the file
        '''
        with open(self.data_file, 'rb') as f:
            if self.data_file.endswith('.gz'):
                raw_text = gzip.decompress(f.read()).decode('utf-8')
            else:
                raw_text = f.read().decode('utf-8')

        return raw_text
        

    def tokenize(self, mode="words"):
        '''
        Create Tokens from the data.

        Args:
        mode (String): Change the mode for the tokenizer.

        Returns:
            List[Strings]: List containing all tokens.
        '''
        if(mode == "words"):
            # ggf. tokenized txt um Satzzeichen bereinigen?
            tokenized_txt = self.load_data().split()
        if(mode == "sentences"):
            tokenized_txt = self.load_data().split(sep='.')
        if(mode == "lines"):
            tokenized_txt = self.load_data().split(sep='\n')

        return tokenized_txt


if __name__ == "__main__":
    dl = DataLoader("test.txt")
    print(f"Raw data: {dl.load_data()}")
    print(f"Word-tokenized: {dl.tokenize(mode='words')}")
    print(f"Sentence-tokenized: {dl.tokenize(mode='sentences')}")