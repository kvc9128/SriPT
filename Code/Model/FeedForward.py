import torch
import torch.nn as nn


class FeedForward(nn.Module):
	def __init__(self, embedding_dimension, feed_forward_dimension):
		super().__init__()
		self.embedding_dimension = embedding_dimension
		self.feed_forward_dimension = feed_forward_dimension
		self.linear_1 = nn.Linear(embedding_dimension, feed_forward_dimension)
		self.linear_2 = nn.Linear(feed_forward_dimension, embedding_dimension)

	def forward(self, x):
		# simple forward pass through linear_layer
		linear_1_out = self.linear_1(x)
		activated = torch.relu(linear_1_out)
		return self.linear_2(activated)
