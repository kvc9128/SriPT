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
		for t in range(3, len(question_tokens)):
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
		logger.info(msg=f"Limited to 2 million sequences due to limited compute")
		shuffled_encoded_sequences = shuffled_encoded_sequences[:2000000]
		shuffled_encoded_targets = shuffled_encoded_targets[:2000000]

	return shuffled_encoded_sequences, shuffled_encoded_targets

def create_sequences_from_common_sense(VOCAB, context_window_length):
	encoded_sequences, encoded_targets = [], []
	tokenized_separator = ["SEP"]
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

				tokenized_question = VOCAB.tokenize_sentence(question_text)
				tokenized_answer = VOCAB.tokenize_sentence(choice_text)
				sentence_encoded_sequences, sentence_encoded_targets = [], []
				if len(tokenized_answer) == 1:
					# single target, so just one sequence is generated
					tokenized_padding = [VOCAB.PAD] * (
								context_window_length - 1 - len(tokenized_question))
					sentence = np.concatenate(
						(tokenized_padding, tokenized_separator, tokenized_question))
					sentence_encoded_sequences.append(
						encode_raw_text(
							sentence,
							VOCAB,
							seq_len=context_window_length
						)
					)
					sentence_encoded_targets.append(
						np.array([VOCAB.word2index(tokenized_answer[0])])
					)
				else:
					# generate multiple sequences based on text
					for t in range(len(tokenized_answer)):
						question_with_part_answer = tokenized_question + tokenized_answer[:t]
						tokenized_padding = [VOCAB.PAD] * (context_window_length - 1 - len(question_with_part_answer))
						sequence = np.concatenate(
							(tokenized_padding, tokenized_separator, question_with_part_answer)
						)
						encoded_sequence = encode_raw_text(
							sequence, VOCAB, seq_len=context_window_length
						)
						encoded_target = np.array([VOCAB.word2index(tokenized_answer[t])])

						sentence_encoded_sequences.append(encoded_sequence)
						sentence_encoded_targets.append(encoded_target)

				sentence_encoded_sequences = np.array(sentence_encoded_sequences)
				sentence_encoded_targets = np.array(sentence_encoded_targets)
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


# def create_sequences_from_squad(VOCAB, context_window_length):
# 	encoded_sequences, encoded_targets = [], []
# 	try:
# 		file_path = "../Datasets/QA/squad.json"
# 		with open(file_path, 'r') as file:
# 			json_data = json.load(file)
#
# 			for item in json_data['data']:
# 				for paragraph in item['paragraphs']:
# 					for qa in paragraph['qas']:
# 						question = qa['question']
# 						# Assumes that you want the text of the first answer (if multiple answers exist)
# 						answer_text = qa['answers'][0]['text'] if qa['answers'] else "No answer"
#
# 						normalized_question = VOCAB.normalize_string(question)
# 						normalized_question = [word for word in normalized_question.split()]
# 						normalized_answer = VOCAB.normalize_string(answer_text)
# 						normalized_answer = [word for word in normalized_answer.split()]
#
# 						sentence = np.concatenate(
# 							(normalized_question, normalized_answer))
# 						sentence_encoded_sequences, sentence_encoded_targets = generate_sequences_for_sentence(
# 							sentence, context_window_length, VOCAB)
#
# 						encoded_sequences.extend(sentence_encoded_sequences)
# 						encoded_targets.extend(sentence_encoded_targets)
#
# 			encoded_sequences = np.array(encoded_sequences)
# 			encoded_targets = np.array(encoded_targets)
# 			indices = np.arange(len(encoded_sequences))
# 			np.random.shuffle(indices)
# 			shuffled_encoded_sequences = encoded_sequences[indices]
# 			shuffled_encoded_targets = encoded_targets[indices]
# 			logger.info(
# 				msg=f"Generated and shuffled {len(shuffled_encoded_sequences)} sequences for SQUAD dataset")
# 			return shuffled_encoded_sequences, shuffled_encoded_targets
#
# 	except FileNotFoundError:
# 		logger.warning(msg="SQUAD not found, returning empty list.")
# 		return np.array([]), np.array([])


def create_sequences_from_squad(VOCAB, context_window_length):
	encoded_sequences, encoded_targets = [], []
	tokenized_separator = ["SEP"]
	try:
		file_path = "../Datasets/QA/squad.json"
		with open(file_path, 'r') as file:
			json_data = json.load(file)

			for item in json_data['data']:
				for paragraph in item['paragraphs']:
					context = paragraph['context']
					tokenized_context = VOCAB.tokenize_sentence(context)

					for qa in paragraph['qas']:
						question = qa['question']
						if qa['answers']:
							answer_text = qa['answers'][0]['text']
						else:
							answer_text = "NO ANSWER"

						tokenized_question = VOCAB.tokenize_sentence(question)
						tokenized_answer = VOCAB.tokenize_sentence(answer_text)

						sentence_encoded_sequences, sentence_encoded_targets = [], []

						if len(tokenized_answer) == 1:
							# Single target, so just one sequence is generated
							tokenized_padding = [VOCAB.PAD] * (
									context_window_length - 1 - len(tokenized_question) - len(tokenized_context)
							)
							sentence = np.concatenate(
								(tokenized_padding, tokenized_context, tokenized_separator,
								 tokenized_question)
							)
							sentence = sentence[-context_window_length:]
							sentence_encoded_sequences.append(
								encode_raw_text(
									sentence,
									VOCAB,
									seq_len=context_window_length
								)
							)
							sentence_encoded_targets.append(
								np.array([VOCAB.word2index(tokenized_answer[0])])
							)
						else:
							# Generate multiple sequences based on text
							for t in range(len(tokenized_answer)):
								context_q_part_answer = tokenized_context + tokenized_separator + tokenized_question + tokenized_answer[:t]
								tokenized_padding = [VOCAB.PAD] * (context_window_length - len(context_q_part_answer))
								sentence = np.concatenate(
									(tokenized_padding, context_q_part_answer)
								)
								sentence = sentence[-context_window_length:]
								encoded_sequence = encode_raw_text(
									sentence, VOCAB, seq_len=context_window_length
								)
								encoded_target = np.array([VOCAB.word2index(tokenized_answer[t])])

								sentence_encoded_sequences.append(encoded_sequence)
								sentence_encoded_targets.append(encoded_target)

						sentence_encoded_sequences = np.array(sentence_encoded_sequences)
						sentence_encoded_targets = np.array(sentence_encoded_targets)
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
		logger.warning(msg="SQuAD not found, returning empty list.")
		return np.array([]), np.array([])

def create_sequences_from_trivia(VOCAB, context_window_length):
	encoded_sequences, encoded_targets = [], []
	tokenized_separator = ["SEP"]
	try:
		file_path = "../Datasets/QA/trivia.json"
		with open(file_path, 'r') as file:
			data = json.load(file)
			# Iterate through each entry in the "Data" list
			for entry in data['Data']:
				question = entry['Question']
				normalized_entity_name = entry['Answer']['Value']
				context = ""  # start with empty context
				descriptions = entry['SearchResults']
				if len(descriptions) > 1:
					# get first context description
					context += (descriptions[0]['Description'])

				tokenized_question = VOCAB.tokenize_sentence(question)
				tokenized_answer = VOCAB.tokenize_sentence(normalized_entity_name)
				tokenized_context = VOCAB.tokenize_sentence(context)

				sentence_encoded_sequences, sentence_encoded_targets = [], []
				if len(tokenized_answer) == 1:
					# single target, so just one sequence is generated
					tokenized_padding = [VOCAB.PAD] * (context_window_length - 1 - len(tokenized_question) - len(tokenized_context))
					sentence = np.concatenate((tokenized_padding, tokenized_context, tokenized_separator, tokenized_question))
					sentence = sentence[-context_window_length:]
					sentence_encoded_sequences.append(
						encode_raw_text(
							sentence,
							VOCAB,
							seq_len=context_window_length
						)
					)
					sentence_encoded_targets.append(
						np.array([VOCAB.word2index(tokenized_answer[0])])
					)
				else:
					# generate multiple sequences based on text
					for t in range(len(tokenized_answer)):
						context_q_part_answer = tokenized_context + tokenized_separator + tokenized_question + tokenized_answer[:t]
						tokenized_padding = [VOCAB.PAD] * (context_window_length - len(context_q_part_answer))
						sentence = np.concatenate(
							(tokenized_padding, context_q_part_answer)
						)
						sentence = sentence[-context_window_length:]
						encoded_sequence = encode_raw_text(
							sentence, VOCAB, seq_len=context_window_length
						)
						encoded_target = np.array([VOCAB.word2index(tokenized_answer[t])])

						sentence_encoded_sequences.append(encoded_sequence)
						sentence_encoded_targets.append(encoded_target)

				sentence_encoded_sequences = np.array(sentence_encoded_sequences)
				sentence_encoded_targets = np.array(sentence_encoded_targets)
				encoded_sequences.extend(sentence_encoded_sequences)
				encoded_targets.extend(sentence_encoded_targets)

			encoded_sequences = np.array(encoded_sequences)
			encoded_targets = np.array(encoded_targets)
			indices = np.arange(len(encoded_sequences))
			np.random.shuffle(indices)
			shuffled_encoded_sequences = encoded_sequences[indices]
			shuffled_encoded_targets = encoded_targets[indices]
			logger.info(
				msg=f"Generated and shuffled {len(shuffled_encoded_sequences)} sequences for TriviaQA dataset")
			return shuffled_encoded_sequences, shuffled_encoded_targets

	except FileNotFoundError:
		logger.warning(msg="TriviaQA dataset not found, returning empty list.")
		return np.array([]), np.array([])


def generate_sequences_for_sentence(normalized_sentence, context_window_length, VOCAB, context=""):
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
