"""
Practical Recurrent Networks in PyTorch
https://www.udemy.com/the-complete-neural-networks-bootcamp-theory-applications/learn/v4/t/lecture/11288834?start=0
"""
import torch


class Dictionary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word: str):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __len__(self):
        return len(self.word2idx)


class TextProcess:
    def __init__(self):
        self.dictionary = Dictionary()

    def get_data(self, path, batch_size=20):
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)
        # Create a 1-D tensor that contains the index of all the words in the file
        rep_tensor = torch.LongTensor(tokens)
        index = 0
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    rep_tensor[index] = self.dictionary.word2idx[word]
                    index += 1
        # Find out how many batches we need
        num_batches = rep_tensor.shape[0] // batch_size
        # Remove the remainder (Filter out the ones that don't fit)
        rep_tensor = rep_tensor[:num_batches * batch_size]
        # return (batch_size, num_batches)
        rep_tensor = rep_tensor.view(batch_size, -1)
        return rep_tensor
