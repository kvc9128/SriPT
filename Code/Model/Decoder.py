import torch.nn as nn
from MultiHeadSelfAttention import MultiHeadedSelfAttention
from FeedForward import FeedForward


class Decoder(nn.Module):
	def __init__(self, embedding_dimension, number_of_heads, feed_forward_dimension, dropout_rate):
		super().__init__()
		self.embedding_dimension = embedding_dimension
		self.num_heads = number_of_heads
		self.feed_forward_dimension = feed_forward_dimension
		self.dropout_rate = dropout_rate

		self.multi_headed_attention = MultiHeadedSelfAttention(
			self.embedding_dimension,
			self.num_heads
		)

		self.dropout_layer = nn.Dropout(p=self.dropout_rate)
		self.feed_forward = FeedForward(embedding_dimension, feed_forward_dimension)
		self.layer_normalization_1 = nn.LayerNorm(embedding_dimension)
		self.layer_normalization_2 = nn.LayerNorm(embedding_dimension)

	def forward(self, x, mask):
		# use the attention layer
		attention_out = self.multi_headed_attention(x, mask)
		# add residual # from ResNet
		attention_output_and_residual = x + attention_out
		# normalize the attention_output_and_residual
		normalized_residual = self.layer_normalization_1(attention_output_and_residual)
		# feed forward
		ff_output = self.feed_forward(normalized_residual)
		# add residual to ff_out
		ff_output_and_residual = normalized_residual + ff_output
		# normalize again
		re_normalized = self.layer_normalization_2(ff_output_and_residual)
		# Dropout, only when training.
		if self.training:
			re_normalized = self.dropout_layer(re_normalized)

		# Residual output
		return re_normalized

	# def original_forward(self, x, mask):
	# 	# normalize the x
	# 	normalized_input = self.layer_normalization_1(x)
	# 	# use the attention layer
	# 	attention_out = self.multi_headed_attention(normalized_input, mask)
	# 	# add residual back
	# 	residual_output = x + attention_out
	# 	# normalization again part 2 electric boogaloo
	# 	re_normalized = self.layer_normalization_2(residual_output)
	# 	# feed forward
	# 	feed_forward_output = self.feed_forward(re_normalized)
	# 	# Dropout, only when training.
	# 	if self.training:
	# 		feed_forward_output = self.dropout_layer(feed_forward_output)
	#
	# 	# Residual output
	# 	return residual_output + feed_forward_output
