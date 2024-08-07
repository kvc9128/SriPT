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


def load_model(model, model_file_name='reuters_base.pt'):
	try:
		if os.path.exists(SAVED_FOLDER):
			model_path = os.path.join(SAVED_FOLDER, model_file_name)
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
	text = text.replace('UNK', '')
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
	text = ' '.join(cleaned_words)
	text = text.replace(' . ', '. ')
	return text


def generate_next_token(model, start_prompt, context_size, VOCAB, max_output_length=32, temperature=25):
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
			output = output / temperature
			output = SOFTMAX(output)

			# Get the last predicted token (the next token)
			# predicted_token_id = torch.argmax(output, dim=-1).item()
			predicted_token_id = torch.multinomial(output, 1).item()

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


def beam_search(model, VOCAB, context_size, beam_size, max_length, start_prompt, temperature):
	"""
	Perform beam search to generate sentences.
	:param temperature:
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
			next_words_prob = next_words_prob / temperature  # apply temperature
			next_words_prob = SOFTMAX(next_words_prob)

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
	return sentence


def setup_generation():
	VOCAB = vocab_manager.load_vocab()
	VOCAB_SIZE = VOCAB.num_words()

	model = SriPT(
		number_of_tokens=VOCAB_SIZE,
		max_sequence_length=CONTEXT_WINDOW,
		embedding_dimension=EMBEDDING_DIMENSION,
		number_of_layers=NUM_DECODER_LAYERS,
		number_of_heads=NUM_ATTENTION_HEADS,
		dropout_rate=DROPOUT_RATE
	)

	# load model and optimizer from previous state
	model = load_model(model)
	prompt = "What is the northern most province in canada"

	total_params = sum(
		param.numel() for param in model.parameters()
	)
	logger.info(f"Model has {total_params} parameters.")

	# spellcheck prompt and get user confirmation
	prompt = spellcheck(prompt)
	print("User prompt:", prompt)
	print("SriPT:", generate_next_token(model, prompt, CONTEXT_WINDOW, VOCAB, temperature=0.01))
	print("SriPT:", generate_next_token(model, prompt, CONTEXT_WINDOW, VOCAB, temperature=0.5))
	print("SriPT:", generate_next_token(model, prompt, CONTEXT_WINDOW, VOCAB, temperature=0.8))
	print("SriPT:", generate_next_token(model, prompt, CONTEXT_WINDOW, VOCAB, temperature=1))
	print("SriPT:", generate_next_token(model, prompt, CONTEXT_WINDOW, VOCAB, temperature=2))

	# experimental beam search - painfully slow
	best_seq = beam_search(model, VOCAB, CONTEXT_WINDOW, beam_size=5,
	                       max_length=15,
	                       start_prompt=prompt,
	                       temperature=0.8)
	print(best_seq)


setup_generation()
