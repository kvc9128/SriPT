import os

import numpy as np
import torch
import logging

from Code import vocab_manager
from Code.Model.SriPT import SriPT
from Code.Utilities.Transcoder import encode_raw_text

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVED_FOLDER = "../TRAINED_MODELS/"


def load_model(model):
	try:
		if os.path.exists(SAVED_FOLDER):
			model_path = os.path.join(SAVED_FOLDER, 'model.pt')
			torch.save(model.state_dict(), model_path)
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


def generate_text(model, start_prompt, context_window, VOCAB, max_output_length=48, temperature=0.95):
	model.eval()
	generated_sequence = start_prompt
	for _ in range(max_output_length):  # don't want to include length of q in a
		# Tokenize the current sequence
		input_ids = encode_raw_text(generated_sequence, VOCAB, context_window, inference=True)
		input_tensor = torch.tensor(np.array([input_ids]), dtype=torch.long, device=DEVICE)

		mask = torch.ones_like(input_tensor)
		mask.to(DEVICE)
		mask[input_tensor == 2] = 0  # 2 is my PAD token

		with torch.no_grad():
			output = model(input_tensor, mask)

		output = output / temperature
		most_likely_tokens = torch.argmax(output, dim=-1)  # pick tokens with the highest probability
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


def generate_next_token(model, start_prompt, context_window, VOCAB, max_output_length=50):
	model.eval()  # Set the model to evaluation mode
	generated_sequence = start_prompt

	for _ in range(max_output_length):
		# Tokenize the current sequence
		input_ids = encode_raw_text(generated_sequence, VOCAB, context_window)
		input_tensor = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)

		mask = torch.ones_like(input_tensor)
		mask.to(DEVICE)
		mask[input_tensor == 2] = 0

		# Generate prediction
		with torch.no_grad():
			output = model(input_tensor, mask)

		# Get the last predicted token (the next token)
		predicted_token_id = torch.argmax(output, dim=-1).item()

		# Break if EOS token is generated
		if predicted_token_id == VOCAB.word2index("EOS"):
			generated_sequence += "."
			break
		else:
			generated_sequence += " " + VOCAB.index2word(predicted_token_id)

	return generated_sequence


def setup_generation():
	# defining hyperparameters TODO: Move these to a file so that we don't need to edit in main and here
	embedding_dimension = 512
	context_window = 64  # context window
	number_of_decoder_layers = 6
	num_attention_heads = 4
	dropout_rate = 0.15
	# wrong file path for vocab for some reason
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
	prompt = "Jason was a"
	# print(generate_text(model, prompt, context_window, VOCAB))
	print(generate_next_token(model, prompt, context_window, VOCAB))


setup_generation()
