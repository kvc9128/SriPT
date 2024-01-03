import logging
import numpy as np

logger = logging.getLogger(__name__)


def encode_raw_text(text, VOCAB, seq_len):
	if isinstance(text, str):
		text = VOCAB.normalize_string(text)
		text = text.split()

	if len(text) > seq_len:
		logger.warning("Truncated sentence, exceeded context window size.")
		text = text[-seq_len:]

	# Original code replaced by more efficient flow below
	# if len(words) < seq_len:
	# 	difference = seq_len - len(words)
	# 	padding = []
	# 	for i in range(difference):
	# 		padding.append(VOCAB.PAD)
	# 	words = padding + words

	padding_length = seq_len - len(text)
	if padding_length > 0:
		padding = [VOCAB.PAD] * padding_length
		text = padding + text

	return np.array([VOCAB.word2index(word) for word in text])  # tokenized_text
