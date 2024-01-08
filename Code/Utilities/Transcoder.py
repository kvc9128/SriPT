import logging
import numpy as np

logger = logging.getLogger(__name__)


def encode_raw_text(text, VOCAB, seq_len, inference=False):
	if isinstance(text, str):
		text = VOCAB.normalize_string(text)
		text = text.split()

	if len(text) > seq_len:
		if not inference:
			logger.debug("Truncated sentence, exceeded context window size.")
		text = text[-seq_len:]

	padding_length = seq_len - len(text)
	if padding_length > 0:
		padding = [VOCAB.PAD] * padding_length
		text = np.concatenate((padding, text))

	return np.array([VOCAB.word2index(word) for word in text])  # tokenized_text
