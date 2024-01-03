import torch

def positional_encoder(embedding_dimensions, sequence_length):
	"""
	Needs to be called only once. These are fixed. Store once and reuse
	:param embedding_dimensions: The number of tokens in our vocabulary
	:param sequence_length: The length of context history we consider. 
	:return: A tensor of shape [seq_len, embedding_dimensions]
	"""
	DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# Like the paper, I will use fixed embeddings, and generate them at run time
	positional_encoding = torch.zeros(sequence_length, embedding_dimensions)
	# position is an array going from 0-seq_len, but reshaped, so it's a 2d matrix [seq_len, 1]
	position = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)
	# for every dimension_index, multiply by 2, divide by embedding_dimensions, and use that as exponent
	# just read the paper again

	div_term = 10000 ** (2 * torch.arange(embedding_dimensions // 2) / (embedding_dimensions // 2))

	# even numbered columns are sine
	positional_encoding[:, 0::2] = torch.sin(position * div_term)
	# odd numbered columns are cosine
	positional_encoding[:, 1::2] = torch.cos(position * div_term)

	positional_encoding = positional_encoding.to(dtype=torch.float, device=DEVICE)
	return positional_encoding
