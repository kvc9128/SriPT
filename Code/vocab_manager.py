import pickle
import json
import argparse
import logging

from vocab import VOCAB
from log_config import setup_logging

# create the logger
logger = logging.getLogger(__name__)
PICKLE_FILE_PATH = "../Datasets/Words/vocab.obj"


def load_books(books: list, vocabulary: VOCAB):
	for book in books:
		vocabulary.add_book_from_txt_file(book)
		logger.debug(msg="Added book " + book[18:])
		file_pi = open(PICKLE_FILE_PATH, 'wb')
		pickle.dump(vocabulary, file_pi)
		logger.debug(msg="Updated pickled file.")
	logger.info(msg="Added all datasets.")


def read_common_sense_qa(vocabulary):
	file_path = "../Datasets/QA/common_sense_q_a.json"
	with open(file_path, 'r') as file:
		for line in file:
			# Parse the JSON object in each line
			json_obj = json.loads(line)

			# Extract words from the question stem
			question_text = json_obj['question']['stem']
			vocabulary.add_raw_sentence(question_text)

			# Get the correct answer's label
			correct_label = json_obj['answerKey']

			# Extract words from the correct choice
			for choice in json_obj['question']['choices']:
				if choice['label'] == correct_label:
					choice_text = choice['text']
					vocabulary.add_raw_sentence(choice_text)
					break  # Exit the loop once the correct choice is found

	file_pi = open(PICKLE_FILE_PATH, 'wb')
	pickle.dump(vocabulary, file_pi)
	logger.info(msg="Wrote common_sense_qa words to pickle")


def read_squad_web_qa(vocabulary):
	file_path = "../Datasets/QA/squad_web.json"
	with open(file_path, 'r') as file:
		data = json.load(file)
		# Iterate through each entry in the "Data" list
		for entry in data['Data']:
			question = entry['Question']
			vocabulary.add_raw_sentence(question)

			normalized_entity_name = entry['Answer']['Value']
			vocabulary.add_raw_sentence(normalized_entity_name)

	file_pi = open(PICKLE_FILE_PATH, 'wb')
	pickle.dump(vocabulary, file_pi)
	logger.info(msg="Wrote squad_web_qa words to pickle")


def read_trivia_qa(vocabulary):
	file_path = "../Datasets/QA/trivia_q_a.json"
	with open(file_path, 'r') as file:
		json_data = json.load(file)

		for item in json_data['data']:
			for paragraph in item['paragraphs']:
				for qa in paragraph['qas']:
					question = qa['question']
					vocabulary.add_raw_sentence(question)
					# Assumes that you want the text of the first answer (if multiple answers exist)
					answer_text = qa['answers'][0]['text'] if qa['answers'] else "No answer"
					vocabulary.add_raw_sentence(answer_text)

	file_pi = open(PICKLE_FILE_PATH, 'wb')
	pickle.dump(vocabulary, file_pi)
	logger.info(msg="Wrote trivia_q_a words to pickle")


def load_vocab():
	try:
		file_path = open(PICKLE_FILE_PATH, 'rb')
		vocabulary = pickle.load(file_path)
		if isinstance(vocabulary, VOCAB):
			logger.debug(msg="vocabulary object found and loaded successfully.")
		else:
			logger.warning(msg="File does not contain a Vocab object. Creating from scratch")
			vocabulary = create_vocab_from_scratch()
		message = f"Vocabulary loaded successfully with {vocabulary.num_words()} words."
		logger.info(msg=message)
		return vocabulary
	except pickle.UnpicklingError:
		logger.critical(msg="UnPickling error. Creating from scratch")
		return create_vocab_from_scratch()
	except FileNotFoundError:
		logger.critical(msg="File does not exist. Creating from scratch")
		return create_vocab_from_scratch()


def create_vocab_from_scratch():
	vocabulary = VOCAB("all_words", min_occurrence=2)
	logger.debug(msg="Read all Unix Words")
	books = [
		"../Datasets/Books/blood_of_olympus.txt",
		"../Datasets/Books/clash_of_kings.txt",
		"../Datasets/Books/house_of_hades.txt",
		"../Datasets/Books/mark_of_athena.txt",
		"../Datasets/Books/percy_jackson_and_the_lightning_thief.txt",
		"../Datasets/Books/storm_of_swords.txt",
		"../Datasets/Books/abbaddons_gate.txt",
		"../Datasets/Books/babylons_ashes.txt",
		"../Datasets/Books/calibans_war.txt",
		"../Datasets/Books/Catcher-in-the-Rye.txt",
		"../Datasets/Books/cibola_burn.txt",
		"../Datasets/Books/jane-austen-pride-prejudice.txt",
		"../Datasets/Books/leviathan_wakes.txt",
		"../Datasets/Books/nemesis_games.txt",
		"../Datasets/Books/persepolis_rising.txt",
		"../Datasets/Books/the_great_gatsby.txt",
		"../Datasets/Books/to_kill_a_mockingbird.txt"
	]

	load_books(books, vocabulary)
	read_common_sense_qa(vocabulary)
	read_trivia_qa(vocabulary)
	read_squad_web_qa(vocabulary)
	vocabulary.enforce_min_count()
	message = f"Vocabulary Created successfully with {vocabulary.num_words()} words."
	logger.info(msg=message)
	return vocabulary


def main():
	parser = argparse.ArgumentParser(description='Process mode.')
	# Add the --mode argument
	parser.add_argument('--mode', default='load', choices=['create', 'load'],
	                    help="Mode of operation: 'create' or 'load'")
	# Parse the arguments
	args = parser.parse_args()

	# Execute based on the mode
	if args.mode == 'load':
		return load_vocab()
	elif args.mode == 'create':
		return create_vocab_from_scratch()


main()
