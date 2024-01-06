"""
This class is responsible for reading and tokenizing data.
This class also is responsible for saving the VOCAB object as a pickled file.

Words/unix-words.txt is treated as the best source for words. It will be trained once
and then stored. After we have updated it with all the words we have in our initial
dataset, we will use it for training the model.
"""

import unicodedata
import re


class VOCAB:
	def __init__(self, name, min_occurrence=3):
		self.SOS = "SOS"
		self.EOS = "EOS"
		self.PAD = "PAD"
		self.UNK = "UNK"
		self.min_count = min_occurrence
		self.write_file_path = "../Datasets/Words/unix-words.txt"
		self.read_file_path = "../Datasets/Words/30k.txt"
		self.name = name  # The name of the vocabulary
		self._word2index = {"SOS": 0, "EOS": 1, "PAD": 2, "UNK": 3}  # Map word to token index
		self._index2word = {0: "SOS", 1: "EOS", 2: "PAD", 3: "UNK"}  # Map token index to word
		# Number of unique words in the corpus
		self._n_words = 4  # Count SOS, EOS and PAD and UNK
		self.add_unix_words()
		self.add_punctuation_and_numbers()

	# Get a list of all words in corpus
	def get_words(self):
		return list(self._word2index.keys())

	# Get the number of words
	def num_words(self):
		return self._n_words

	# Convert a word into a token index
	def word2index(self, word):
		if word not in self._word2index:
			return self._word2index[self.PAD]
		else:
			return self._word2index[word]

	# Convert a token into a word
	def index2word(self, token):
		if token not in self._index2word:
			return self._index2word[self._word2index[self.PAD]]
		else:
			return self._index2word[token]

	# Add all words from 30k-words to VOCAB object that occur min_occurrence number of times
	def add_unix_words(self):
		word_count = {}
		with open(self.read_file_path, 'r') as file:
			words = file.readlines()
		# Remove newline characters
		for word in words:
			word = word.strip()
			if word in word_count:
				word_count[word] += 1
			else:
				word_count[word] = 1
		for word, count in word_count.items():
			if count >= self.min_count:
				self.add_word(word)

	# Add all the words in a sentence to the vocabulary
	def add_normalized_sentence(self, sentence):
		for word in sentence.split(' '):
			self.add_word(word)

	# meant to be called once
	def add_punctuation_and_numbers(self):
		for i in range(10):
			self._word2index[str(i)] = self._n_words
			self._index2word[self._n_words] = str(i)
			self._n_words += 1
		self.add_normalized_sentence(". , ' ; : > < / ? ! # $ ^ * ( ) { } [ ]")

	# Add a single word to the vocabulary
	def add_word(self, word):
		if word not in self._word2index:
			self._word2index[word] = self._n_words
			self._index2word[self._n_words] = word
			self._n_words += 1
			with open(self.write_file_path, 'a') as file:
				# Write each new word on a new line
				file.write(word + '\n')

	@staticmethod
	def unicode_to_ascii(s):
		return ''.join(
			c for c in unicodedata.normalize('NFD', s)
			if unicodedata.category(c) != 'Mn'
		)

	@staticmethod
	def normalize_string(s):
		s = VOCAB.unicode_to_ascii(s.lower().strip())
		s = re.sub(r"([?!]+)", r"\1 ", s)
		s = re.sub(r"\.", " EOS ", s)  # Replace periods with 'EOS'
		s = re.sub(r"\?", " EOS ", s)  # Replace Question marks with 'EOS'
		s = re.sub(r"!", " EOS ", s)  # Replace Exclamation marks with 'EOS'
		s = re.sub(r"[^a-zA-ZEOS!?]+", r" ", s)
		return s

	def add_book_from_txt_file(self, filename):
		with open(filename, encoding='utf-8') as f:
			text = f.read()

		# Convert any unicode to ascii, and normalize the string
		normalized_text = VOCAB.normalize_string(text)
		self.add_normalized_sentence(normalized_text)

	def add_raw_sentence(self, s):
		normalized_text = VOCAB.normalize_string(s)
		self.add_normalized_sentence(normalized_text)
