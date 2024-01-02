import torch.nn as nn


class LMHead(nn.Module):
	"""
    a linear layer that maps the embedding dimension to the vocabulary size.
    """

	def __init__(self, embedding_dimension, number_of_tokens):
		super().__init__()
		self.embedding_dimension = embedding_dimension
		self.number_of_tokens = number_of_tokens
		self.linear = nn.Linear(embedding_dimension, number_of_tokens)

	def forward(self, x):
		"""
	    x dimensions are: (batch_size, sequence_length, embedding_dimension)
	    output dimensions are: (batch_size, sequence_length, number_of_tokens)

	    This version
	    """
		# Compute the linear layer
		# linear_output dimensions are: (batch_size, sequence_length, number_of_tokens)
		return self.linear(x)

	# def next_token_prediction_forward(self, x):
	# 	# predict next token only
	# 	last_position_output = x[:, -1, :]
	# 	linear_output = self.linear(last_position_output)
	# 	return linear_output
