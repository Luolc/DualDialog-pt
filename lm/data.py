import os
import torch
import pickle


class Dictionary(object):
    def __init__(self):
        self.word2idx = {'<unk>': 0}
        self.idx2word = ['<unk>']
        self.wordcnt = {'<unk>': 1}

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
            self.wordcnt[word] = 1
        else:
            self.wordcnt[word] += 1
        return self.word2idx[word]

    def getid(self, word, thresh=4):
        if (word not in self.word2idx) or (self.wordcnt[word] < thresh):
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, bptt):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'), bptt)
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'), bptt)
        self.test = self.tokenize(os.path.join(path, 'test.txt'), bptt)

        with open(os.path.join(path, 'dict.pkl'), 'wb+') as f:
            pickle.dump(self.dictionary, f)

    def tokenize(self, path, bptt):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            tokens = 0
            for line in f:
                line = line.lower()
                line = line.strip()
                n_tokens = len(line.split())
                if n_tokens >= bptt:
                    continue
                words = line.split() + ['<eos>'] + ['<pad>'] * (bptt - n_tokens - 1)
                assert len(words) == bptt
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                line = line.lower()
                line = line.strip()
                n_tokens = len(line.split())
                if n_tokens >= bptt:
                    continue
                words = line.split() + ['<eos>'] + ['<pad>'] * (bptt - n_tokens - 1)
                for word in words:
                    ids[token] = self.dictionary.getid(word)
                    token += 1

        return ids
