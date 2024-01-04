import json
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
			logger.info(f"Found {len(normalized_text)} words in book")
			# TODO change how much txt we are passing in, only reduce for testing
			return generate_sequences_for_text(
				normalized_text,
				context_window_length,
				VOCAB,
				text_file_path
			)

	except FileNotFoundError:
		logger.warning(msg="File not found, returning empty list.")
		return np.array([]), np.array([])


def create_sequences_from_json(VOCAB, json_file_path, context_window_length):
	if json_file_path == "../Datasets/QA/common_sense_q_a.json":
		return create_sequences_from_common_sense(VOCAB, context_window_length)
	elif json_file_path == "../Datasets/QA/trivia_q_a.json":
		return create_sequences_from_trivia(VOCAB, context_window_length)
	elif json_file_path == "../Datasets/QA/squad_web.json":
		return create_sequences_from_SQuAD(VOCAB, context_window_length)


def generate_sequences_for_text(normalized_text, context_window_length, VOCAB, text_file_path):
	"""
	example
	sequence = [PAD, PAD, I, AM, PERCY] # irl tokens(ints), used words for ease of understanding
	target = [JACKSON]

	:param normalized_text:
	:param context_window_length:
	:param VOCAB:
	:param text_file_path:
	:return:
	"""
	encoded_sequences, encoded_targets = generate_sequences_old(normalized_text,
	                                                            context_window_length,
	                                                            VOCAB)
	indices = np.arange(len(encoded_sequences))
	np.random.shuffle(indices)
	shuffled_encoded_sequences = encoded_sequences[indices]
	shuffled_encoded_targets = encoded_targets[indices]
	# Since we have such a large set of datasets, limit each book to 2 million sequences
	if len(shuffled_encoded_sequences) > 2e6:
		shuffled_encoded_sequences = shuffled_encoded_sequences[:int(2e6)]
		shuffled_encoded_targets = shuffled_encoded_targets[:int(2e6)]
	logger.info(
		msg=f"Generated and shuffled {len(shuffled_encoded_sequences)} sequences for book " + text_file_path[
		                                                                                      18:])
	return shuffled_encoded_sequences, shuffled_encoded_targets


def create_sequences_from_common_sense(VOCAB, context_window_length):
	encoded_sequences, encoded_targets = [], []
	try:
		file_path = "../Datasets/QA/common_sense_q_a.json"
		with open(file_path, 'r') as file:
			for line in file:
				# Parse the JSON object in each line
				json_obj = json.loads(line)

				# Extract words from the question stem
				question_text = json_obj['question']['stem']
				# Get the correct answer's label
				correct_label = json_obj['answerKey']
				# Extract words from the correct choice
				choice_text = ""
				for choice in json_obj['question']['choices']:
					if choice['label'] == correct_label:
						choice_text += choice['text']
						break  # Exit the loop once the correct choice is found

				normalized_question = VOCAB.normalize_string(question_text)
				normalized_question = [word for word in normalized_question.split()]
				normalized_choice = VOCAB.normalize_string(choice_text)
				normalized_choice = [word for word in normalized_choice.split()]

				sentence = np.concatenate((normalized_question, normalized_choice))
				sentence_encoded_sequences, sentence_encoded_targets = generate_sequences_for_sentence(
					sentence,
					context_window_length,
					VOCAB
					)

				encoded_sequences.extend(sentence_encoded_sequences)
				encoded_targets.extend(sentence_encoded_targets)

			encoded_sequences = np.array(encoded_sequences)
			encoded_targets = np.array(encoded_targets)
			indices = np.arange(len(encoded_sequences))
			np.random.shuffle(indices)
			shuffled_encoded_sequences = encoded_sequences[indices]
			shuffled_encoded_targets = encoded_targets[indices]
			logger.info(
				msg=f"Generated and shuffled {len(shuffled_encoded_sequences)} sequences for common sense dataset")
			return shuffled_encoded_sequences, shuffled_encoded_targets

	except FileNotFoundError:
		logger.warning(msg="CommonSenseQA not found, returning empty list.")
		return np.array([]), np.array([])


def create_sequences_from_trivia(VOCAB, context_window_length):
	encoded_sequences, encoded_targets = [], []
	try:
		file_path = "../Datasets/QA/trivia_q_a.json"
		with open(file_path, 'r') as file:
			json_data = json.load(file)

			for item in json_data['data']:
				for paragraph in item['paragraphs']:
					for qa in paragraph['qas']:
						question = qa['question']
						# Assumes that you want the text of the first answer (if multiple answers exist)
						answer_text = qa['answers'][0]['text'] if qa['answers'] else "No answer"

						normalized_question = VOCAB.normalize_string(question)
						normalized_question = [word for word in normalized_question.split()]
						normalized_answer = VOCAB.normalize_string(answer_text)
						normalized_answer = [word for word in normalized_answer.split()]

						sentence = np.concatenate(
							(normalized_question, normalized_answer))
						sentence_encoded_sequences, sentence_encoded_targets = generate_sequences_for_sentence(
							sentence,
							context_window_length,
							VOCAB
						)

						encoded_sequences.append(sentence_encoded_sequences)
						encoded_targets.append(sentence_encoded_targets)

			encoded_sequences = np.array(encoded_sequences)
			encoded_targets = np.array(encoded_targets)
			indices = np.arange(len(encoded_sequences))
			np.random.shuffle(indices)
			shuffled_encoded_sequences = encoded_sequences[indices]
			shuffled_encoded_targets = encoded_targets[indices]
			logger.info(
				msg=f"Generated and shuffled {len(shuffled_encoded_sequences)} sequences for trivia dataset")
			return shuffled_encoded_sequences, shuffled_encoded_targets

	except FileNotFoundError:
		logger.warning(msg="Trivia QA not found, returning empty list.")
		return np.array([]), np.array([])


def create_sequences_from_SQuAD(VOCAB, context_window_length):
	encoded_sequences, encoded_targets = [], []
	try:
		file_path = "../Datasets/QA/squad_web.json"
		with open(file_path, 'r') as file:
			data = json.load(file)
			# Iterate through each entry in the "Data" list
			for entry in data['Data']:
				question = entry['Question']
				normalized_entity_name = entry['Answer']['Value']

				normalized_question = VOCAB.normalize_string(question)
				normalized_question = [word for word in normalized_question.split()]
				normalized_answer = VOCAB.normalize_string(normalized_entity_name)
				normalized_answer = [word for word in normalized_answer.split()]

				sentence = np.concatenate(
					(normalized_question, normalized_answer))
				sentence_encoded_sequences, sentence_encoded_targets = generate_sequences_for_sentence(
					sentence,
					context_window_length,
					VOCAB
				)

				encoded_sequences.append(sentence_encoded_sequences)
				encoded_targets.append(sentence_encoded_targets)

			encoded_sequences = np.array(encoded_sequences)
			encoded_targets = np.array(encoded_targets)
			indices = np.arange(len(encoded_sequences))
			np.random.shuffle(indices)
			shuffled_encoded_sequences = encoded_sequences[indices]
			shuffled_encoded_targets = encoded_targets[indices]
			logger.info(
				msg=f"Generated and shuffled {len(shuffled_encoded_sequences)} sequences for SQuAD dataset")
			return shuffled_encoded_sequences, shuffled_encoded_targets

	except FileNotFoundError:
		logger.warning(msg="SQuAD dataset not found, returning empty list.")
		return np.array([]), np.array([])


def generate_sequences_old(normalized_text, context_window_length, VOCAB):
	"""
	This is the old format of sequence generation, for example
	sequence = [PAD, PAD, I, AM, PERCY] # irl tokens(ints), used words for ease of understanding
	target = [JACKSON]

	:param normalized_text:
	:param context_window_length:
	:param VOCAB:
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
	return encoded_sequences, encoded_targets


def generate_sequences_for_sentence(normalized_sentence, context_window_length, VOCAB):
	normalized_sentence = normalized_sentence[-context_window_length:]  # crop long sentences
	encoded_sequences, encoded_targets = [], []
	for t in range(1, len(normalized_sentence) - 1):
		encoded_sequence = encode_raw_text(normalized_sentence[:t],
		                                   VOCAB,
		                                   seq_len=context_window_length)
		encoded_target = np.array([VOCAB.word2index(normalized_sentence[t + 1])])

		encoded_sequences.append(encoded_sequence)
		encoded_targets.append(encoded_target)
	encoded_sequences, encoded_targets = np.array(encoded_sequences), np.array(encoded_targets)
	return encoded_sequences, encoded_targets
