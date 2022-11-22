import os
import re
import sys
import string
import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(0)


class LangDataset(Dataset):
    """
    Define a pytorch dataset class that accepts a text path, and optionally label path and
    a vocabulary (depends on your implementation). This class holds all the data and implement
    a __getitem__ method to be used by a Python generator object or other classes that need it.

    DO NOT shuffle the dataset here, and DO NOT pad the tensor here.
    """
    def __init__(self, text_path, label_path=None, vocab=None):
        """
        Read the content of vocab and text_file
        Args:
            vocab (string): Path to the vocabulary file.
            text_file (string): Path to the text file.
        """
        # texts
        with open(text_path, encoding='utf-8') as f:
            self.texts = f.read().splitlines()
            

        # labels + vocab
        self.labels = None
        if label_path != None:
            self.text_vocab = {"padding": 0}
            for text in self.texts:
                self.create_text_vocab(text, self.text_vocab)
            self.label_vocab = {}
            with open(label_path, encoding='utf-8') as f:
                self.labels = f.read().splitlines()
                self.create_label_vocab(self.labels, self.label_vocab)
        else:
            assert vocab != None , "Testing must use vocab from training"
            checkpoint = torch.load(vocab)
            self.text_vocab = checkpoint["text_vocab"]
            self.label_vocab = checkpoint["label_vocab"]
            
    def create_text_vocab(self, data, vocab):
        data = [*data]
        bigrams = [i + j for i, j in zip(data, data[1:])]
        for bigram in bigrams:
            if bigram not in vocab:
                vocab[bigram] = len(vocab)

    
    def create_label_vocab(self, data, vocab):
        for label in data:
            if label not in vocab:
                vocab[label] = len(vocab)
                
                
    def make_bow_vector(self, data):
        num_vocab, num_class = self.vocab_size()
        x = []
        data = [*data]
        bigrams = [i + j for i, j in zip(data, data[1:])]
        for bigram in bigrams:
            if bigram in self.text_vocab:
                x.append(self.text_vocab[bigram])
            else:
                x.append(0)
        return torch.LongTensor(x)
                
    
    def make_target(self, label):
        return torch.LongTensor([self.label_vocab[label]])
        

    def vocab_size(self):
        """
        A function to inform the vocab size. The function returns two numbers:
            num_vocab: size of the vocabulary
            num_class: number of class labels
        """
        num_vocab = len(self.text_vocab)
        num_class = len(self.label_vocab)
        return num_vocab, num_class
    
    def __len__(self):
        """
        Return the number of instances in the data
        """
        return len(self.texts)

    def __getitem__(self, i):
        """
        Return the i-th instance in the format of:
            (text, label)
        Text and label should be encoded according to the vocab (word_id).

        DO NOT pad the tensor here, do it at the collator function.
        """
        text = self.make_bow_vector(self.texts[i])
        label = torch.empty(1)
        if self.labels != None:
            label = self.make_target(self.labels[i])
        
        return text, label


class Model(nn.Module):
    """
    Define a model that with one embedding layer with dimension 16 and
    a feed-forward layers that reduce the dimension from 16 to 200 with ReLU activation
    a dropout layer, and a feed-forward layers that reduce the dimension from 200 to num_class
    """
    def __init__(self, num_vocab, num_class, dropout=0.3):
        super().__init__()
        # define your model here
        self.embedding = nn.Embedding(num_vocab, 16, padding_idx=0) # 0 padding will remain as 0
        self.ff1 = nn.Sequential(nn.Linear(16, 200), nn.ReLU())
        self.ff2 = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(200, num_class))

    def forward(self, x):
        # define the forward function here
        h0 = self.embedding(x)
        # h0 = mean of embedding layer
        mask = h0.sum(dim=-1) != 0
        h0 = h0.sum(dim=-2) / mask.sum(dim=1).view(-1, 1)
        # feed forward
        h1 = self.ff1(h0)
        h2 = self.ff2(h1)
        p = F.log_softmax(h2, dim=1)
        return p


def collator(batch):
    """
    Define a function that receives a list of (text, label) pair
    and return a pair of tensors:
        texts: a tensor that combines all the text in the mini-batch, pad with 0
        labels: a tensor that combines all the labels in the mini-batch
    """
    texts = []
    labels = []
    for text, label in batch:
        texts.append(text)
        labels.append(label)
    return torch.nn.utils.rnn.pad_sequence(texts, batch_first=True), torch.cat(labels)


def train(model, dataset, batch_size, learning_rate, num_epoch, device='cpu', model_path=None):
    """
    Complete the training procedure below by specifying the loss function
    and optimizers with the specified learning rate and specified number of epoch.
    
    Do not calculate the loss from padding.
    """
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, shuffle=True)

    # assign these variables
    criterion = nn.NLLLoss() # equivalent to cross entropy when paired with log softmax
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    start = datetime.datetime.now()
    for epoch in range(num_epoch):
        model.train()
        running_loss = 0.0
        for step, data in enumerate(data_loader, 0):
            # get the inputs; data is a tuple of (inputs, labels)
            texts = data[0].to(device)
            labels = data[1].to(device)

            # zero the parameter gradients
            model.zero_grad()
            
            # do forward propagation
            log_probs = model.forward(texts)
            
            # do loss calculation
            loss = criterion(log_probs, labels)
            
            # do backward propagation
            loss.backward()

            # do parameter optimization step
            optimizer.step()
            
            # calculate running loss value for non padding
            running_loss += loss.item()

            # print loss value every 100 steps and reset the running loss
            if step % 100 == 99:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, step + 1, running_loss / 100))
                running_loss = 0.0

    end = datetime.datetime.now()
    
    # define the checkpoint and save it to the model path
    # tip: the checkpoint can contain more than just the model
    checkpoint = {
                  "text_vocab": dataset.text_vocab,
                  "label_vocab": dataset.label_vocab,
                  "epoch": epoch,
                  "model_state_dict": model.state_dict(),
                  "optimizer_state_dict": optimizer.state_dict(),
                  "loss": criterion,
                 }
    torch.save(checkpoint, model_path)

    print('Model saved in ', model_path)
    print('Training finished in {} minutes.'.format((end - start).seconds / 60.0))


def test(model, dataset, class_map, device='cpu'):
    model.eval()
    data_loader = DataLoader(dataset, batch_size=20, collate_fn=collator, shuffle=False)
    labels = []
    with torch.no_grad():
        for data in data_loader:
            texts = data[0].to(device)
            outputs = model(texts).cpu()
            # get the label predictions
            for log_probs in outputs:
                labels.append(class_map[torch.argmax(log_probs.squeeze()).item()])
    return labels


def main(args):
    if torch.cuda.is_available():
        device_str = 'cuda:{}'.format(0)
    else:
        device_str = 'cpu'
    device = torch.device(device_str)
    
    assert args.train or args.test, "Please specify --train or --test"
    if args.train:
        assert args.label_path is not None, "Please provide the labels for training using --label_path argument"
        dataset = LangDataset(args.text_path, args.label_path)
        num_vocab, num_class = dataset.vocab_size()
        model = Model(num_vocab, num_class).to(device)
        
        # you may change these hyper-parameters
        learning_rate = 0.1
        batch_size = 10
        num_epochs = 100

        train(model, dataset, batch_size, learning_rate, num_epochs, device, args.model_path)
    if args.test:
        assert args.model_path is not None, "Please provide the model to test using --model_path argument"
        
        # create the test dataset object using LangDataset class
        dataset = LangDataset(args.text_path, vocab=args.model_path)

        # initialize and load the model
        num_vocab, num_class = dataset.vocab_size()
        model = Model(num_vocab, num_class).to(device)
        model.load_state_dict(torch.load(args.model_path)["model_state_dict"])

        # the lang map should contain the mapping between class id to the language id (e.g. eng, fra, etc.)
        lang_map = list(dataset.label_vocab.keys())

        # run the prediction
        preds = test(model, dataset, lang_map, device)
        
        # write the output
        with open(args.output_path, 'w', encoding='utf-8') as out:
            out.write('\n'.join(preds))
    print('\n==== A2 Part 2 Done ====')


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_path', help='path to the text file')
    parser.add_argument('--label_path', default=None, help='path to the label file')
    parser.add_argument('--train', default=False, action='store_true', help='train the model')
    parser.add_argument('--test', default=False, action='store_true', help='test the model')
    parser.add_argument('--model_path', required=True, help='path to the output file during testing')
    parser.add_argument('--output_path', default='out.txt', help='path to the output file during testing')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    main(args)