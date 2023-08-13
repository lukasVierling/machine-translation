import pickle
import sys

class PickleLoader:
    """
    Wrapper for loading and saving objects to pickle files
    """
    @staticmethod
    def load(path: str) -> object:
        """
        Loads an object from a given pickle file and returns it

        Args:
            path (str): The path to the pickle file to load, relative to the cwd
        """
        with open(path, 'rb') as f:
            return pickle.load(f)

    @staticmethod  
    def save(path: str, obj: object) -> None:
        """
        Saves an object to a given pickle file

        Args:
            path (str): The path to the pickle file to save, relative to the cwd
            obj (object): The object to save
        """
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

if __name__ == "__main__":
    # Test the PickleLoader
    test_obj = {"a": 1, "b": 2, "c": 3}
    PickleLoader.save("data/test.pkl", test_obj)
    loaded_obj = PickleLoader.load("data/test.pkl")
    print(loaded_obj)