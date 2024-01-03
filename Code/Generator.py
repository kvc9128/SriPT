import os
import torch
import logging
import torch.nn.functional as F

from Code import vocab_manager
from Code.Model.SriPT import SriPT
from Code.Utilities.Transcoder import encode_raw_text

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_FILE_PATH = "TRAINED_MODELS/checkpoint.pth"


def load_model(model):
	try:
		if os.path.exists(MODEL_FILE_PATH):
			checkpoint = torch.load(MODEL_FILE_PATH)
			model.load_state_dict(checkpoint['model_state_dict'])
			logger.info(f"Successfully loaded model with weights and parameters")
		else:
			logger.error("No checkpoint found. Please provide a model and checkpoint.")
	except FileNotFoundError:
		logger.error(f"Checkpoint file not found: {MODEL_FILE_PATH}")
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
	start_prompt_len = len(start_prompt.split())
	for _ in range(max_output_length + start_prompt_len):  # don't want to include length of q in a
		# Tokenize the current sequence
		input_ids = encode_raw_text(generated_sequence, VOCAB, context_window)
		input_tensor = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)

		mask = torch.ones_like(input_tensor)
		mask.to(DEVICE)
		mask[input_tensor == 2] = 0  # 2 is my PAD token

		with torch.no_grad():
			output = model(input_tensor, mask)

		output = output / temperature
		probabilities = F.softmax(output, dim=-1)

		predicted_token_id = torch.multinomial(probabilities, 1).item()

		if predicted_token_id == VOCAB.word2index("EOS"):
			generated_sequence += "."
			break
		else:
			generated_sequence += " " + VOCAB.index2word(predicted_token_id)

	return generated_sequence


def setup_generation():
	# defining hyperparameters TODO: Move these to a file so that we don't need to edit in main and here
	embedding_dimension = 256
	context_window = 24  # context window
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
	prompt = "His joints hurt"
	print(generate_text(model, prompt, context_window, VOCAB))


setup_generation()
