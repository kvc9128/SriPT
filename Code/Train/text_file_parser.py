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
			# convert to word tokens (still english)
			normalized_words = VOCAB.tokenize_sentence(text)
			logger.info(f"Found ~{len(normalized_words)} words in book")
			return generate_sequences_for_text(
				normalized_words,
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
	elif json_file_path == "../Datasets/QA/squad.json":
		return create_sequences_from_squad(VOCAB, context_window_length)
	elif json_file_path == "../Datasets/QA/trivia.json":
		return create_sequences_from_trivia(VOCAB, context_window_length)


def generate_sequences_for_text(normalized_text, context_window_length, VOCAB, text_file_path, step=256):
	encoded_sequences, encoded_targets = [], []
	sep_token = "SEP"
	pad_token = "PAD"

	for i in range(0, len(normalized_text) - 1 - context_window_length, step):
		question_tokens = normalized_text[i: i + context_window_length]
		target_tokens = normalized_text[i + 1: i + context_window_length + 1]
		for t in range(0, len(question_tokens)):
			sequence = [pad_token] * (context_window_length - 2 - t) + [sep_token] + question_tokens[:t + 1]
			target = target_tokens[t]

			encoded_sequence = encode_raw_text(sequence, VOCAB, context_window_length, inference=False)
			encoded_target = np.array([VOCAB.word2index(target)])

			encoded_sequences.append(encoded_sequence)
			encoded_targets.append(encoded_target)

	encoded_sequences, encoded_targets = np.array(encoded_sequences), np.array(encoded_targets)
	indices = np.arange(len(encoded_sequences))
	np.random.shuffle(indices)
	shuffled_encoded_sequences = encoded_sequences[indices]
	shuffled_encoded_targets = encoded_targets[indices]
	logger.info(
		msg=f"Generated and shuffled {len(shuffled_encoded_sequences)} sequences for book " + text_file_path[
		                                                                                      18:])
	if len(shuffled_encoded_sequences) > 2e6:
		logger.info(msg=f"Limited to 2 million sequences for compute reasons")
		shuffled_encoded_sequences = shuffled_encoded_sequences[:2000000]
		shuffled_encoded_targets = shuffled_encoded_targets[:2000000]

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


def create_sequences_from_squad(VOCAB, context_window_length):
	encoded_sequences, encoded_targets = [], []
	try:
		file_path = "../Datasets/QA/squad.json"
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

						encoded_sequences.extend(sentence_encoded_sequences)
						encoded_targets.extend(sentence_encoded_targets)

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


def create_sequences_from_trivia(VOCAB, context_window_length):
	encoded_sequences, encoded_targets = [], []
	try:
		file_path = "../Datasets/QA/trivia.json"
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

				encoded_sequences.extend(sentence_encoded_sequences)
				encoded_targets.extend(sentence_encoded_targets)

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
