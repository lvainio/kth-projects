import numpy as np
import requests
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


class Data:

    def __init__(self, sequence_length=100, batch_size=64, train_split=0.7, val_split=0.1, test_split=0.2):
        url = "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
        url = "https://www.gutenberg.org/cache/epub/73614/pg73614.txt"
        url = "https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt"
        self.text = self._download_text(url)

        self.vocabulary = sorted(set(self.text))
        self.vocabulary_size = len(self.vocabulary) 

        self.char_to_index = {char: index for index, char in enumerate(self.vocabulary)}
        self.index_to_char = np.array(self.vocabulary)
        
        text_as_indices = torch.tensor([self.char_to_index[char] for char in self.text]) 
        num_samples = len(text_as_indices) // (sequence_length + 1) 

        inputs = torch.reshape(text_as_indices[:num_samples*sequence_length], (-1, sequence_length))
        outputs = torch.reshape(text_as_indices[1:1+num_samples*sequence_length], (-1, sequence_length))

        dataset = TensorDataset(inputs, outputs)
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_split, val_split, test_split])

        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    def _download_text(self, url):
        response = requests.get(url)
        if response.status_code == 200:
            text = response.text
        else:
            raise Exception("Could not download shakespeare.txt")
        return text

        
def main():
    Data()


if __name__ == "__main__":
    main()




