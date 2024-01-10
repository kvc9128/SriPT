import os

import numpy as np
import torch
import logging

import torch.nn as nn

from textblob import TextBlob
from Code import vocab_manager
from Code.Model.SriPT import SriPT
from model_hyperparameters import *
from Code.Utilities.Transcoder import encode_raw_text

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOFTMAX = nn.Softmax(dim=-1)
SAVED_FOLDER = "../TRAINED_MODELS/"


def load_model(model):
	try:
		if os.path.exists(SAVED_FOLDER):
			model_path = os.path.join(SAVED_FOLDER, 'model.pt')
			model.load_state_dict(torch.load(model_path, map_location=torch.device(DEVICE)))
			logger.info(f"Successfully loaded model with weights and parameters")
		else:
			logger.error("No checkpoint found. Please provide a model and checkpoint.")
	except FileNotFoundError:
		logger.error(f"Checkpoint file not found: {SAVED_FOLDER}")
	except KeyError as e:
		logger.error(f"Missing key in checkpoint data: {e}")
	except RuntimeError as e:
		logger.error(f"Error loading state dict: {e}")
	except Exception as e:  # Generic catch-all for other exceptions
		logger.error(f"An error occurred while loading the model: {e}")
	finally:
		return model

def remove_repeated_phrases(text):
	words = text.split()
	seen = set()
	cleaned_words = []

	for word in words:
		if word not in seen:
			seen.add(word)
			cleaned_words.append(word)
		else:
			if all(previous_word != word for previous_word in cleaned_words[-len(seen):]):
				cleaned_words.append(word)

	return ' '.join(cleaned_words)


def generate_text_from_seq_2_seq_model(model, start_prompt, context_size, VOCAB, max_output_length=48,
                                       temperature=0.95):
	model.eval()
	generated_sequence = start_prompt
	for _ in range(max_output_length):  # don't want to include length of q in a
		# Tokenize the current sequence
		input_ids = encode_raw_text(generated_sequence, VOCAB, context_size, inference=True)
		input_tensor = torch.tensor(np.array([input_ids]), dtype=torch.long, device=DEVICE)

		mask = torch.ones_like(input_tensor)
		mask.to(DEVICE)
		mask[input_tensor == 2] = 0  # 2 is my PAD token

		with torch.no_grad():
			output = model(input_tensor, mask)

		output = output / temperature
		most_likely_tokens = torch.argmax(output,
		                                  dim=-1)  # pick tokens with the highest probability
		most_likely_tokens = most_likely_tokens[0]  # get the last token
		most_likely_token = most_likely_tokens[-1]

		if int(most_likely_token) == VOCAB.word2index("EOS"):
			generated_sequence += "."
			break
		elif int(most_likely_token) == VOCAB.word2index("PAD"):
			generated_sequence += ""
		else:
			generated_sequence += " " + VOCAB.index2word(int(most_likely_token))

	return generated_sequence


def generate_next_token(model, start_prompt, context_size, VOCAB, max_output_length=32, temperature=0.9):
	model.eval()  # Set the model to evaluation mode
	generated_sequence = start_prompt

	for _ in range(max_output_length):
		# Tokenize the current sequence
		input_ids = np.array([encode_raw_text(generated_sequence, VOCAB, context_size)])
		input_tensor = torch.tensor(input_ids, dtype=torch.long, device=DEVICE)

		mask = torch.ones_like(input_tensor)
		mask.to(DEVICE)
		mask[input_tensor == 2] = 0  # 2 is my PAD token index

		# Generate prediction
		with torch.no_grad():
			output = model(input_tensor, mask)
			output = SOFTMAX(output)
			output = output / temperature

		# Get the last predicted token (the next token)
		predicted_token_id = torch.argmax(output, dim=-1).item()

		# Break if EOS token is generated
		if predicted_token_id == VOCAB.word2index("EOS") or predicted_token_id == VOCAB.word2index(
				"eos"):
			generated_sequence += "."
		# break
		else:
			generated_sequence += " " + VOCAB.index2word(predicted_token_id)

	return remove_repeated_phrases(generated_sequence)


def spellcheck(prompt):
	corrected_prompt = TextBlob(prompt)
	# We think there are some spelling errors, offer user a chance to re-enter or proceed
	if prompt != corrected_prompt:
		print("We found a few spelling mistakes in your prompt, here is the corrected prompt")
		print("If you meant to type what you did, please enter 1, or 2 to use autocorrected text")
		print("Prompt\t1.", prompt)
		print("Corrected Prompt\t2.", corrected_prompt)
		choice = input("Your choice, (1) or (2) =")
		if choice == "1":
			return prompt
		elif choice == "2":
			return corrected_prompt
		else:
			print("Invalid choice, using given prompt")
			return prompt
	else:
		return prompt


def beam_search(model, VOCAB, context_size, beam_size, max_length, start_prompt):
	"""
	Perform beam search to generate sentences.
	:param context_size:
	:param VOCAB:
	:param start_prompt:
	:param model: Trained language model
	:param beam_size: Number of beams to keep at each step
	:param max_length: Maximum length of the generated sequence
	:return: Best generated sequence
	"""
	length_penalty_alpha = 0.7
	# Starting the beam search with the initial token
	prompt_tokens = np.array(encode_raw_text(start_prompt, VOCAB, context_size))
	start_token = prompt_tokens[-1]

	start_beam = ([start_token], 1.0)  # sequence, probability
	beams = [start_beam]

	for _ in range(max_length):
		candidates = []
		for seq, seq_prob in beams:
			if seq[-1] == 1:  # EOS token is 1
				candidates.append([seq, seq_prob])
				continue

			# Get probabilities of next words for the current sequence
			input_tensor = torch.tensor([seq], dtype=torch.long, device=DEVICE)
			mask = torch.ones_like(input_tensor)
			mask.to(DEVICE)
			mask[input_tensor == 2] = 0  # 2 is my PAD token index
			next_words_prob = model(input_tensor, mask)
			next_words_prob = SOFTMAX(next_words_prob)
			next_words_prob = next_words_prob / 0.9  # apply temperature

			token_prob_pairs = sorted(enumerate(next_words_prob[0]), key=lambda x: x[1],
			                          reverse=True)

			for token_idx, prob in token_prob_pairs[:beam_size]:
				new_seq = seq + [token_idx]
				new_length = len(new_seq)
				new_prob = (seq_prob * prob) ** (1 / (new_length ** length_penalty_alpha))
				candidates.append([new_seq, new_prob])

		# Sort all candidates by probability and select top 'beam_size' sequences
		candidates.sort(key=lambda x: x[1], reverse=True)
		beams = candidates[:beam_size]

	# Choose the sequence with the highest probability
	best_sequence = max(beams, key=lambda x: x[1])[0]
	# convert tokens to sentence
	sentence = start_prompt + " "
	best_sequence = best_sequence[1:]  # ignore the starter token
	for token in best_sequence:
		sentence += VOCAB.index2word(token) + " "
	sentence = sentence[:-1]
	sentence += "."
	return remove_repeated_phrases(sentence)


def setup_generation():
	VOCAB = vocab_manager.load_vocab()
	VOCAB_SIZE = VOCAB.num_words()

	model = SriPT(
		number_of_tokens=VOCAB_SIZE,
		max_sequence_length=context_window,
		embedding_dimension=embedding_dimension,
		number_of_layers=number_of_decoder_layers,
		number_of_heads=num_attention_heads,
		dropout_rate=dropout_rate
	)

	# load model and optimizer from previous state
	model = load_model(model)
	prompt = "Who is Poseidon's son?"

	# print(generate_text(model, prompt, context_size, VOCAB))

	total_params = sum(
		param.numel() for param in model.parameters()
	)
	logger.info(f"Model has {total_params} parameters.")

	# spellcheck prompt and get user confirmation
	prompt = spellcheck(prompt)
	print(generate_next_token(model, prompt, context_window, VOCAB))

	# experimental beam search - painfully slow
	best_seq = beam_search(model, VOCAB, context_window, beam_size=3, max_length=15, start_prompt=prompt)
	print(best_seq)


setup_generation()
