import torch.nn as nn
from Code.Model.Decoder import Decoder


class DecoderStack(nn.Module):
	def __init__(
			self,
			embedding_dimension,
			number_of_layers,
			number_of_heads,
			feed_forward_dimension,
			dropout_rate,
			max_sequence_length
	):
		super().__init__()
		self.embedding_dimension = embedding_dimension
		self.number_of_layers = number_of_layers
		self.number_of_heads = number_of_heads
		self.feed_forward_dimension = feed_forward_dimension
		self.dropout_rate = dropout_rate
		self.max_sequence_length = max_sequence_length

		# Create the decoder layers
		self.decoder_layers = nn.ModuleList([
			Decoder(
				embedding_dimension,
				number_of_heads,
				feed_forward_dimension,
				dropout_rate
			)
			for _ in range(number_of_layers)
		])

	def forward(self, x, mask):
		decoder_out = x
		for decoder_layer in self.decoder_layers:
			decoder_out = decoder_layer(decoder_out, mask)

		return decoder_out
