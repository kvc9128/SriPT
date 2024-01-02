import logging
import numpy as np
from Code.Utilities.Transcoder import encode_raw_text

logger = logging.getLogger(__name__)


def create_sequences_from_book(VOCAB, text_file_path, context_window_length):
	try:
		with open(text_file_path, encoding='utf-8') as f:
			text = f.read()
			# Convert any unicode to ascii, and normalize the string
			normalized_text = VOCAB.normalize_string(text)
			normalized_text = [word for word in normalized_text.split()]

			return generate_sequences_for_text(
				normalized_text,
				context_window_length,
				VOCAB,
				text_file_path
			)

	except FileNotFoundError:
		logger.warning(msg="File not found, returning empty list.")
		return np.array([]), np.array([])


def generate_sequences_for_text(normalized_text, context_window_length, VOCAB, text_file_path):
	"""
	This is the new format of sequence generation, This should allow us to encode QA pairs.
	This function however, only deals with

	for example
	sequence = [PAD, PAD, I, AM, PERCY] # irl tokens(ints), used words for ease of understanding
	target = [JACKSON]

	:param normalized_text:
	:param context_window_length:
	:param VOCAB:
	:param text_file_path:
	:return:
	"""
	encoded_sequences, encoded_targets = [], []
	for i in range(0, len(normalized_text) - 1 - context_window_length):
		sequence = normalized_text[i: i + context_window_length]
		target = normalized_text[i + 1: i + context_window_length + 1]
		for t in range(context_window_length):
			encoded_sequence = encode_raw_text(
				sequence[:t + 1], VOCAB,
				seq_len=context_window_length
			)
			encoded_target = encode_raw_text(
				target[t], VOCAB,
				seq_len=context_window_length
			)

			encoded_sequences.append(encoded_sequence)
			encoded_targets.append(encoded_target)

	encoded_sequences, encoded_targets = np.array(encoded_sequences), np.array(encoded_targets)
	indices = np.arange(len(encoded_sequences))
	np.random.shuffle(indices)
	shuffled_encoded_sequences = encoded_sequences[indices]
	shuffled_encoded_targets = encoded_targets[indices]
	logger.info(msg="Generated and shuffled sequences for book " + text_file_path)
	return shuffled_encoded_sequences, shuffled_encoded_targets


def generate_sequences_old(normalized_text, context_window_length, VOCAB, text_file_path):
	"""
	This is the old format of sequence generation, for example
	sequence = [PAD, PAD, I, AM, PERCY] # irl tokens(ints), used words for ease of understanding
	target = [JACKSON]

	:param normalized_text:
	:param context_window_length:
	:param VOCAB:
	:param text_file_path:
	:return:
	"""
	encoded_sequences, encoded_targets = [], []
	for i in range(0, len(normalized_text) - 1 - context_window_length):
		sequence = normalized_text[i: i + context_window_length]
		target = normalized_text[i + 1: i + context_window_length + 1]
		for t in range(context_window_length):
			encoded_sequence = encode_raw_text(
				sequence[:t + 1], VOCAB,
				seq_len=context_window_length
			)
			encoded_target = np.array([VOCAB.word2index(target[t])])

			encoded_sequences.append(encoded_sequence)
			encoded_targets.append(encoded_target)

	encoded_sequences, encoded_targets = np.array(encoded_sequences), np.array(encoded_targets)
	indices = np.arange(len(encoded_sequences))
	np.random.shuffle(indices)
	shuffled_encoded_sequences = encoded_sequences[indices]
	shuffled_encoded_targets = encoded_targets[indices]
	logger.info(msg="Generated and shuffled sequences for book " + text_file_path)
	return shuffled_encoded_sequences, shuffled_encoded_targets
