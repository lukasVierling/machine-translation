class Normalizer():
    
    def __remove_capital_letters(self, text: str) -> str:
        """
        Remove capital letters from the given text.

        Args:
        - text (str): The input text to be normalized.

        Returns:
        - str: The normalized text with all capital letters removed.
        """
        normalized_text = text
        return normalized_text

    def __remove_contractions(self, text: str) -> str:
        """
        Remove contractions from the given text.

        Args:
        - text (str): The input text to be normalized.

        Returns:
        - str: The normalized text with all contractions expanded.
        """
        normalized_text = text
        return normalized_text

    def __remove_special_characters(self, text: str) -> str:
        """
        Remove special characters from the given text.

        Args:
        - text (str): The input text to be normalized.

        Returns:
        - str: The normalized text with all special characters removed.
        """
        normalized_text = text
        return normalized_text

    def __remove_punctuation(self, text: str) -> str:
        """
        Remove punctuation from the given text.

        Args:
        - text (str): The input text to be normalized.

        Returns:
        - str: The normalized text with all punctuation marks removed.
        """
        normalized_text = text
        return normalized_text

    def normalize_data(self, text: str, 
                       capital_letters: bool=True, 
                       contractions: bool=True, 
                       special_characters: bool=True, 
                       punctuation: bool=True) -> str:
        """
        Normalize the given text according to the specified normalization options.

        Args:
        - text (str): The input text to be normalized.
        - capital_letters (bool): If True, remove all capital letters from the text.
        - contractions (bool): If True, expand all contractions in the text.
        - special_characters (bool): If True, remove all special characters from the text.
        - punctuation (bool): If True, remove all punctuation marks from the text.

        Returns:
        - str: The normalized text according to the specified normalization options.
        """
        normalized_text = text
        if(capital_letters):
            normalized_text = self.__remove_capital_letters(normalized_text)
        if(contractions):
            normalized_text = self.__remove_contractions(normalized_text)
        if(special_characters):
            normalized_text = self.__remove_special_characters(normalized_text)
        if(punctuation):
            normalized_text = self.__remove_punctuation(normalized_text)
        return normalized_text