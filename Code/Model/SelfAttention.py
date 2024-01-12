import torch
import torch.nn as nn
import numpy as np


class SelfAttention(nn.Module):  # This class is the self attention layer
	def __init__(self, embedding_dimension, head_dimension):
		super().__init__()
		self.embedding_dimension = embedding_dimension
		self.head_dimension = head_dimension
		self.q = nn.Linear(self.embedding_dimension, self.head_dimension)
		self.k = nn.Linear(self.embedding_dimension, self.head_dimension)
		self.v = nn.Linear(self.embedding_dimension, self.head_dimension)

		self.softmax = nn.Softmax(dim=-1)

	def forward(self, x, mask):
		"""
	    Input is of shape [batch size, sequence_length, EMBEDDING_DIMENSION]
	    mask is to help us ignore the pad tokens and to white-out anything we haven't seen yet.

	    mask is of shape batch_size, sequence length
	    """
		query = self.q(x)
		key = self.k(x)
		value = self.v(x)

		attention_weights = torch.matmul(query, key.transpose(-2, -1))  # =QK.T
		attention_weights = attention_weights / np.sqrt(self.head_dimension)  # =(QK.T)/√head_dim

		# Apply the mask to the attention weights, by setting the masked tokens to a very low value.
		# This will make the softmax output 0 for these values.
		mask = mask.reshape(attention_weights.shape[0], 1, attention_weights.shape[2])
		attention_weights = attention_weights.masked_fill(mask == 2, 1e-9)  # 2 is pad for us

		# Softmax makes sure all scores are between 0 and 1 and the sum of scores is 1.
		# attention_scores dimensions are: (batch_size, sequence_length, sequence_length)
		attention_scores = self.softmax(attention_weights)

		# Output dimensions are: (batch_size, sequence_length, head_dimension)
		return torch.bmm(attention_scores, value)  # =((QK.T)/√head_dim)*V
