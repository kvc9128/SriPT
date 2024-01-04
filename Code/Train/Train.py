import os
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
from Code.vocab_manager import load_vocab
from Code.Train.text_file_parser import create_sequences_from_book
from Code.Train.text_file_parser import create_sequences_from_json

logger = logging.getLogger(__name__)
QUICK_SAVE_PATH = "quicksave/"
SAVE_PATH = "../TRAINED_MODELS/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Train:
	def __init__(
			self,
			model,
			batch_size,
			num_epochs,
			loss_fn,
			optimizer,
			context_window
	):
		self.train_losses = []
		self.encoded_targets = None
		self.encoded_sequences = None
		self.VOCAB = load_vocab()
		self.batch_size = batch_size
		self.num_epochs = num_epochs
		self.loss_fn = loss_fn
		self.optimizer = optimizer
		self.model = model
		self.context_window = context_window

	# Training loop
	def train(self):
		# Internal function - do not use directly
		train_losses = []
		# Preallocate tensors
		sequence_tensor = torch.empty((self.batch_size, self.context_window), dtype=torch.long,
		                              device=DEVICE)
		mask_tensor = torch.empty_like(sequence_tensor, dtype=torch.long, device=DEVICE)
		target_tensor = torch.empty((self.batch_size, 1), dtype=torch.long,
		                            device=DEVICE)

		for epoch in range(self.num_epochs):
			losses = []
			for i in range(0, len(self.encoded_sequences), self.batch_size):
				if i + self.batch_size > len(self.encoded_sequences):
					break  # Skip the last partial batch

				batch_sequences = self.encoded_sequences[i:i + self.batch_size]
				batch_targets = self.encoded_targets[i:i + self.batch_size]

				sequence_tensor[:len(batch_sequences)].copy_(torch.tensor(batch_sequences,
				                                                          dtype=torch.long,
				                                                          device=DEVICE))
				target_tensor[:len(batch_targets)].copy_(torch.tensor(batch_targets,
				                                                      dtype=torch.long,
				                                                      device=DEVICE))
				mask_tensor[:len(batch_sequences)].fill_(1)
				mask_tensor[sequence_tensor == self.VOCAB.word2index("PAD")] = 0

				# Forward pass
				model_out = self.model(sequence_tensor, mask_tensor)
				target_tensor_squeezed = target_tensor.squeeze(1)

				# Forward pass (Seq2Seq style)
				# model_out = self.model(sequence_tensor, mask_tensor)
				# model_out = model_out.view(-1, model_out.size(-1))
				# target_tensor_squeezed = target_tensor.view(-1)

				# Backpropagation
				loss = self.loss_fn(model_out, target_tensor_squeezed)
				loss.backward()
				self.optimizer.step()
				self.optimizer.zero_grad()

				losses.append(loss.item())

			epoch_loss = np.average(losses)
			train_losses.append(epoch_loss)
			logger.info('\tEpoch: ' + str(epoch) + ' Loss: ' + str(epoch_loss))

		self.train_losses.extend(train_losses)
		return self.model, self.optimizer

	def plot_loss(self):
		plt.plot(self.train_losses)
		plt.yscale('log')
		plt.ylabel('Loss')
		plt.xlabel('Epoch')
		plt.show()

	@staticmethod
	def quick_save_model(model, optimizer, document, path):
		os.makedirs(path, exist_ok=True)
		# save the model
		model_path = os.path.join(path, 'model.pt')
		torch.save(model.state_dict(), model_path)
		optimizer_path = os.path.join(path, 'optimizer.pt')
		torch.save(optimizer.state_dict(), optimizer_path)

		logger.info(
			msg=f"Stored model for document {document}")

	def update_model_and_optimizer(self, model, optimizer):
		self.optimizer = optimizer
		self.model = model
		logger.debug(f"Successfully updated model and optimizer")

	def train_model_on(self, data_file_path):
		"""
		:param data_file_path:
		:return:
		"""
		logger.info(f"Started training on {data_file_path}")
		encoded_sequences, encoded_targets = None, None
		if data_file_path[-4:] == ".txt":
			encoded_sequences, encoded_targets = create_sequences_from_book(
				VOCAB=self.VOCAB,
				text_file_path=data_file_path,
				context_window_length=self.context_window
			)
			logger.debug("Obtained sequences for book")

		elif data_file_path[-5:] == ".json":
			encoded_sequences, encoded_targets = create_sequences_from_json(
				VOCAB=self.VOCAB,
				json_file_path=data_file_path,
				context_window_length=self.context_window
			)
			logger.debug(f"Obtained sequences for json file {data_file_path}")

		self.encoded_sequences = encoded_sequences
		self.encoded_targets = encoded_targets
		model, optimizer = self.train()
		self.plot_loss()
		Train.quick_save_model(model, optimizer, data_file_path, SAVE_PATH)
		logger.info(f"Successfully trained on {data_file_path}")
