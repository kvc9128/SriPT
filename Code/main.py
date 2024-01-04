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
	"../Datasets/Books/Cracking-the-Coding-Interview.txt",
	"../Datasets/Books/Data Mining Concepts and Techniques.txt",
	"../Datasets/Books/Deep Learning by Ian Goodfellow.txt",
	"../Datasets/Books/Elements of Statistical Learning.txt",
	"../Datasets/Books/house_of_hades.txt",
	"../Datasets/Books/MachineLearning by TomMitchell.txt",
	"../Datasets/Books/mark_of_athena.txt",
	"../Datasets/Books/percy_jackson_and_the_greek_gods.txt",
	"../Datasets/Books/percy_jackson_and_the_lightning_thief.txt",
	"../Datasets/Books/storm_of_swords.txt",
	"../Datasets/QA/squad_web.json",
	"../Datasets/QA/trivia_q_a.json",
	"../Datasets/QA/common_sense_q_a.json"
]
SAVED_FOLDER = "../TRAINED_MODELS/"


def load_model(model, optimizer):
	try:
		if os.path.exists(SAVED_FOLDER):
			model_path = os.path.join(SAVED_FOLDER, 'model.pt')
			torch.save(model.state_dict(), model_path)
			optimizer_path = os.path.join(SAVED_FOLDER, 'optimizer.pt')
			torch.save(optimizer.state_dict(), optimizer_path)
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
		return model, optimizer


def main():
	# Hyperparameters
	embedding_dimension = 256
	context_window = 32  # context window
	number_of_decoder_layers = 6
	num_attention_heads = 4
	dropout_rate = 0.15
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
	OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=0.0001)
	LOSS_FN = nn.CrossEntropyLoss()
	EPOCHS = 6
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

	dataset = datasets[-3]  # common sense
	trainer.train_model_on(dataset)

	MODEL, OPTIMIZER = load_model(MODEL, OPTIMIZER)
	trainer.update_model_and_optimizer(MODEL, OPTIMIZER)
	dataset = datasets[-2]  # trivia
	trainer.train_model_on(dataset)

	MODEL, OPTIMIZER = load_model(MODEL, OPTIMIZER)
	trainer.update_model_and_optimizer(MODEL, OPTIMIZER)
	dataset = datasets[-1]  # squad
	trainer.train_model_on(dataset)

	MODEL, OPTIMIZER = load_model(MODEL, OPTIMIZER)
	trainer.update_model_and_optimizer(MODEL, OPTIMIZER)
	dataset = datasets[0]  # blood of olympus
	trainer.train_model_on(dataset)

	MODEL, OPTIMIZER = load_model(MODEL, OPTIMIZER)
	trainer.update_model_and_optimizer(MODEL, OPTIMIZER)
	dataset = datasets[1]  # clash of kings
	trainer.train_model_on(dataset)



main()
