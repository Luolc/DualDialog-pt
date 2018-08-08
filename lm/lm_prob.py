# -*- coding: utf-8 -*-

import math
import torch
import pickle


class LMProb(object):
    def __init__(self, model_path, dict_path):
        with open(model_path, 'rb') as f:
            self.model = torch.load(f)
            self.model.eval()
            self.model = self.model.cpu()

        with open(dict_path, 'rb') as f:
            self.dictionary = pickle.load(f)

    def get_prob(self, words, verbose=False):
        words = words + ['<eos>']
        ids = [self.dictionary.getid(w) for w in words]
        input = torch.LongTensor([ids[0]]).unsqueeze(0)

        if verbose:
            print('words =', words)
            print('ids =', ids)

        hidden = self.model.init_hidden(1)
        log_probs = []

        with torch.no_grad():
            for i in range(1, len(words)):
                output, hidden = self.model(input, hidden)
                word_weights = output.squeeze().data.exp()

                prob = word_weights[ids[i]] / word_weights.sum()
                log_probs.append(math.log(prob))
                input.data.fill_(int(ids[i]))

        if verbose:
            for i in range(len(log_probs)):
                print('  {} => {:d},\tlogP(w|s)={:.4f}'.format(words[i + 1], ids[i + 1],
                                                               log_probs[i]))
            print('\n  => sum_prob = {:.4f}'.format(sum(log_probs)))

        return sum(log_probs) / math.sqrt(len(log_probs))


if __name__ == '__main__':
    sample_words = ['i', 'love', 'you', '.']
    lmprob = LMProb('save/daily.pt', 'data/daily/dict.pkl')
    norm_prob = lmprob.get_prob(sample_words, verbose=True)
    print('\n  => norm_prob = {:.4f}'.format(norm_prob))
