import os
import torch
import logging
import torch.nn as nn
import vocab_manager
from Model.SriPT import SriPT
from Train.Train import Train

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
datasets = [
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
	"../Datasets/QA/squad_web.json",
	"../Datasets/QA/trivia_q_a.json",
	"../Datasets/QA/common_sense_q_a.json"
]
SAVED_FOLDER = "../TRAINED_MODELS/"


def load_model(model, optimizer):
	try:
		if os.path.exists(SAVED_FOLDER):
			model_path = os.path.join(SAVED_FOLDER, 'model.pt')
			model.load_state_dict(torch.load(model_path, map_location=torch.device(DEVICE)))
			optimizer_path = os.path.join(SAVED_FOLDER, 'optimizer.pt')
			optimizer.load_state_dict(torch.load(optimizer_path, map_location=torch.device(DEVICE)))
			logger.info(f"Successfully saved model with weights and parameters")
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
		return model, optimizer


def main():
	# Hyperparameters
	embedding_dimension = 512
	context_window = 32  # context window
	number_of_decoder_layers = 8
	num_attention_heads = 4
	dropout_rate = 0.1
	VOCAB = vocab_manager.load_vocab()
	VOCAB_SIZE = VOCAB.num_words()
	logger.info(msg=f"Running on {DEVICE}")
	MODEL = SriPT(
		number_of_tokens=VOCAB_SIZE,
		max_sequence_length=context_window,
		embedding_dimension=embedding_dimension,
		number_of_layers=number_of_decoder_layers,
		number_of_heads=num_attention_heads,
		dropout_rate=dropout_rate
	)
	MODEL.to(DEVICE)
	OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=0.0005)
	LOSS_FN = nn.CrossEntropyLoss()
	EPOCHS = 40
	BATCH_SIZE = 64

	trainer = Train(
		model=MODEL,
		batch_size=BATCH_SIZE,
		num_epochs=EPOCHS,
		loss_fn=LOSS_FN,
		optimizer=OPTIMIZER,
		context_window=context_window
	)

	total_params = sum(
		param.numel() for param in MODEL.parameters()
	)
	logger.info(f"Model has {total_params} parameters.")

	for dataset in datasets:
		trainer.train_model_on(dataset)
		MODEL, OPTIMIZER = load_model(MODEL, OPTIMIZER)
		trainer.update_model_and_optimizer(MODEL, OPTIMIZER)


main()
