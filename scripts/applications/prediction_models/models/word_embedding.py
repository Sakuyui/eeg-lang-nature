import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List
import itertools
class WordEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim = 128, context_size = 2):
        super(WordEmbeddingModel, self).__init__()
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.vocal_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)
    
    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs
    
    def train(self, sentences, epoch = 10, lr = 0.001):
        losses = []
        loss_function = nn.NLLLoss()
        model = self
        optimizer = optim.SGD(model.parameters(), lr=0.001)
        sentences = list(itertools.chain.from_iterable(sentences))
        ngrams = [
            (
                [sentences[i - j - 1] for j in range(self.context_size)],
                sentences[i]
            )
            for i in range(self.context_size, len(sentences))
        ]
        for epoch in range(epoch):
            total_loss = 0
            print("epoch = %d" % epoch)
            for context, target in ngrams:
                # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
                # into integer indices and wrap them in tensors)
                context_idxs = torch.tensor(context, dtype=torch.long)

                # Step 2. Recall that torch *accumulates* gradients. Before passing in a
                # new instance, you need to zero out the gradients from the old
                # instance
                model.zero_grad()

                # Step 3. Run the forward pass, getting log probabilities over next
                # words
                log_probs = model(context_idxs)

                # Step 4. Compute your loss function. (Again, Torch wants the target
                # word wrapped in a tensor)

                loss = loss_function(log_probs, torch.tensor([target], dtype=torch.long))

                # Step 5. Do the backward pass and update the gradient
                loss.backward()
                optimizer.step()

                # Get the Python number from a 1-element Tensor by calling tensor.item()
                total_loss += loss.item()
            losses.append(total_loss)
            print("loss = %lf" % total_loss)
  

        