import torch.nn as nn

from LM_Head import LMHead
from DecoderStack import DecoderStack
from Code.Utilities.Positional_Encoder import positional_encoder


class SriPT(nn.Module):
	def __init__(
			self,
			number_of_tokens,  # The number of tokens in the vocabulary
			max_sequence_length=64,  # The context window
			embedding_dimension=512,  # The dimension of the token embeddings
			number_of_layers=6,  # The number of decoder layers to use
			number_of_heads=4,  # The number of attention heads to use
			feed_forward_dimension=1024,  # The dimension of the feed forward layer
			dropout_rate=0.1  # The dropout rate to use
	):
		super().__init__()
		self.number_of_tokens = number_of_tokens
		self.max_sequence_length = max_sequence_length
		self.embedding_dimension = embedding_dimension
		self.number_of_layers = number_of_layers
		self.number_of_heads = number_of_heads

		if feed_forward_dimension is None:
			self.feed_forward_dimension = embedding_dimension * 4
		else:
			self.feed_forward_dimension = feed_forward_dimension

		self.dropout_rate = dropout_rate

		# Create the token embedding layer
		self.token_embedding = nn.Embedding(
			num_embeddings=number_of_tokens,
			embedding_dim=embedding_dimension
		)

		# Create the normalization layer
		self.layer_normalization = nn.LayerNorm(embedding_dimension)

		# Create the decoder stack
		self.decoder = DecoderStack(
			embedding_dimension=embedding_dimension,
			number_of_layers=number_of_layers,
			number_of_heads=number_of_heads,
			feed_forward_dimension=self.feed_forward_dimension,
			dropout_rate=dropout_rate,
			max_sequence_length=max_sequence_length
		)

		# Create the language model head
		self.lm_head = LMHead(embedding_dimension, number_of_tokens)
		self.positional_encoding = positional_encoder(
			self.embedding_dimension,
			sequence_length=max_sequence_length
		)

	def forward(self, x, mask):
		# token_embeddings dimensions are: (batch_size, sequence_length, embedding_dimension)
		token_embeddings = self.token_embedding(x)
		# positional_encoding dimensions are: (batch_size, sequence_length, embedding_dimension)
		position_encoded_tokens = self.positional_encoding[:token_embeddings.size(1), :] + token_embeddings
		# Post embedding layer normalization
		positional_encoding_normalized = self.layer_normalization(position_encoded_tokens)
		decoder_outputs = self.decoder(positional_encoding_normalized, mask)
		lm_head_outputs = self.lm_head(decoder_outputs)

		return lm_head_outputs
