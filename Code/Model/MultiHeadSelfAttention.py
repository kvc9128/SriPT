import torch
import torch.nn as nn
from Code.Model.SelfAttention import SelfAttention


class MultiHeadedSelfAttention(nn.Module):
	def __init__(self, embedding_dimension, number_of_heads):
		super().__init__()
		self.embedding_dimension = embedding_dimension
		self.head_dimension = embedding_dimension // number_of_heads
		self.number_of_heads = number_of_heads
		self.multi_heads = nn.ModuleList([SelfAttention(embedding_dimension, self.head_dimension) for _ in range(number_of_heads)])

		# Create a linear layer to combine the outputs of the self attention modules
		self.output_layer = nn.Linear(number_of_heads * self.head_dimension, embedding_dimension)

	def forward(self, x, mask):
		self_attention_outputs = []
		for head in self.multi_heads:
			output = head(x, mask)
			self_attention_outputs.append(output)
		concatenated_self_attention_outputs = torch.cat(self_attention_outputs, dim=2)
		return self.output_layer(concatenated_self_attention_outputs)
