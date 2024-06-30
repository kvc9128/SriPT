"""
This class is responsible for reading and tokenizing data.
This class also is responsible for saving the VOCAB object as a pickled file.

Words/unix-words.txt is treated as the best source for words. It will be trained once
and then stored. After we have updated it with all the words we have in our initial
dataset, we will use it for training the model.
"""

import re
import nltk
import unicodedata

from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


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
		self._word2index = {"SOS": 0, "EOS": 1, "PAD": 2, "UNK": 3, "SEP": 4}  # Map word to token index
		self._index2word = {0: "SOS", 1: "EOS", 2: "PAD", 3: "UNK", 4: "SEP"}  # Map token index to word
		self._word_count = {"SOS": self.min_count,
		                    "EOS": self.min_count,
		                    "PAD": self.min_count,
		                    "UNK": self.min_count,
		                    "SEP": self.min_count}  # keep track of word count to only keep words with min_occurrence
		# Number of unique words in the corpus
		self._n_words = 5  # Count SOS, EOS, PAD, SEP and UNK
		# Add digits and decimal point
		self.add_digits()
		# Add/remove words from vocab object
		self.add_unix_words()

	def add_digits(self):
		for i in range(10):
			self.add_word(str(i))
		self.add_word(".")

	def enforce_min_count(self):
		"""
		At the end of this function, we will have words in unix-txt that will not be a part of
		this model

		this deletes a few numbers
		:return:
		"""
		for word, count in self._word_count.items():
			if count < self.min_count:
				token = self.word2index(word)
				# delete from word-token map
				del self._word2index[word]
				# delete token from token-word map
				del self._index2word[token]
		self._n_words = len(list(self._word2index.keys()))
		new_word2index, new_index2word = {}, {}
		new_idx = 0
		# Ensure that it is a continuous set of numbers from 0-_n_words
		for index in self._index2word.keys():
			new_index2word[new_idx] = self._index2word[index]
			new_word2index[self._index2word[index]] = new_idx
			new_idx += 1
		self._index2word = new_index2word
		self._word2index = new_word2index

	# Get a list of all words in corpus
	def get_words(self):
		return list(self._word2index.keys())

	# Get the number of words
	def num_words(self):
		return self._n_words

	# Convert a word into a token index
	def word2index(self, word):
		if word not in self._word2index:
			return self._word2index[self.UNK]
		else:
			return self._word2index[word]

	# Convert a token into a word
	def index2word(self, token):
		if token not in self._index2word:
			return self._index2word[self._word2index[self.UNK]]
		else:
			return self._index2word[token]

	# Add all words from 30k-words to VOCAB object that occur min_occurrence number of times
	def add_unix_words(self):
		with open(self.read_file_path, 'r') as file:
			words = file.readlines()
		# Remove newline characters
		for word in words:
			word = word.strip()
			if word in self._word_count:
				self._word_count[word] += 1
			else:
				self._word_count[word] = 1

			self.add_word(word)

	# Add all the words in a sentence to the vocabulary
	def add_normalized_sentence(self, sentence):
		for word in VOCAB.tokenize_sentence(sentence):
			self.add_word(word)

	# Add a single word to the vocabulary
	def add_word(self, word):
		if word not in self._word2index:
			self._word2index[word] = self._n_words
			self._index2word[self._n_words] = word
			self._n_words += 1
			self._word_count[word] = 1
			with open(self.write_file_path, 'a') as file:
				# Write each new word on a new line
				file.write(word + '\n')
		else:
			self._word_count[word] += 1

	@staticmethod
	def unicode_to_ascii(s):
		return ''.join(
			c for c in unicodedata.normalize('NFD', s)
			if unicodedata.category(c) != 'Mn'
		)

	@staticmethod
	def normalize_string(s):
		s = VOCAB.unicode_to_ascii(s.lower().strip())
		s = s.replace('\n', ' ').replace('\t', '')
		s = re.sub(r'[^\w\s.]', '', s)

		words = word_tokenize(s)
		return ' '.join(words)

	def add_book_from_txt_file(self, filename):
		with open(filename, encoding='utf-8') as f:
			text = f.read()

		# Convert any unicode to ascii, and normalize the string
		normalized_text = VOCAB.normalize_string(text)
		self.add_normalized_sentence(normalized_text)

	def add_raw_sentence(self, s):
		normalized_text = VOCAB.normalize_string(s)
		self.add_normalized_sentence(normalized_text)

	@staticmethod
	def tokenize_sentence(sentence):
		"""
		Sentence is a string. Given a string "I have 200 apples and 2.5 oranges."
		This function will return ['I', 'have', '2', '0', '0', 'apples', 'and', '2', '.', '5', 'oranges']

		The array will contain actual human-readable words. They will need to be tokenized using
		the word2index function
		:param sentence:
		:return:
		"""
		# normalize string first
		normalized_text = VOCAB.normalize_string(sentence)
		# Split to handle numbers correctly
		words = []
		for word in normalized_text.split(' '):
			if re.match(r'^\d+(\.\d+)?$', word):  # Check if the word is a number (int or float)
				for char in word:
					words.append(char)
			else:
				words.append(word)
		return words
