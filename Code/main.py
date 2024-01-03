import torch
import logging
import torch.nn as nn
import vocab_manager
from Model.SriPT import SriPT
from Train.Train import Train

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
books = [
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
	"../Datasets/Books/storm_of_swords.txt"
]


def main():
	# Hyperparameters
	embedding_dimension = 256
	context_window = 24  # context window
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
	EPOCHS = 4
	BATCH_SIZE = 256

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
	book = books[0]  # change manually as we train
	logger.info(f"Model has {total_params} parameters in model")
	# print(MODEL)
	# trainer.train_model_on(book)


main()
