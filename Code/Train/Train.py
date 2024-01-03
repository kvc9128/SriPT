import os
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
from Code.vocab_manager import load_vocab
from Code.Train.text_file_parser import create_sequences_from_book

logger = logging.getLogger(__name__)
QUICK_SAVE_PATH = "quicksave/"
MODEL_FILE_PATH = "../TRAINED_MODELS/"


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
		sequence_tensor = torch.empty((self.batch_size, self.context_window), dtype=torch.long)
		mask_tensor = torch.empty_like(sequence_tensor, dtype=torch.long)
		target_tensor = torch.empty((self.batch_size, self.context_window), dtype=torch.long)

		for epoch in range(self.num_epochs):
			losses = []
			for i in range(0, len(self.encoded_sequences), self.batch_size):
				if i + self.batch_size > len(self.encoded_sequences):
					break  # Skip the last partial batch

				batch_sequences = self.encoded_sequences[i:i + self.batch_size]
				batch_targets = self.encoded_targets[i:i + self.batch_size]

				sequence_tensor[:len(batch_sequences)].copy_(torch.tensor(batch_sequences,
				                                                          dtype=torch.long))
				target_tensor[:len(batch_targets)].copy_(torch.tensor(batch_targets,
				                                                      dtype=torch.long))
				mask_tensor[:len(batch_sequences)].fill_(1)
				mask_tensor[sequence_tensor == self.VOCAB.word2index("PAD")] = 0

				# Forward pass
				# model_out = self.model(sequence_tensor, mask_tensor)
				# target_tensor_squeezed = target_tensor.squeeze(1)

				# Forward pass (Seq2Seq style)
				model_out = self.model(sequence_tensor, mask_tensor)
				model_out = model_out.view(-1, model_out.size(-1))
				target_tensor_squeezed = target_tensor.view(-1)
				loss = self.loss_fn(model_out, target_tensor_squeezed)

				# Backpropagation
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
	def quick_save_model(model, optimizer, chunk_idx, document, path):
		checkpoint = {
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'chunk_id': chunk_idx
		}
		os.makedirs(path, exist_ok=True)
		torch.save(checkpoint, os.path.join(path, 'checkpoint.pth'))
		with open(os.path.join(path, 'last_index.txt'), 'w') as f:
			f.write(str(chunk_idx))
		logger.info(msg=f"Stored model and optimizer after {chunk_idx}e6 sequences for document {document}")

	def load_quick_save(self):
		"""
		The expected flow is every so often, we save our progress. If there is an issue,
		we must manually use load_quick_save to restore progress instead of starting from
		scratch on a book.
		:return:
		"""
		checkpoint_file = os.path.join(QUICK_SAVE_PATH, 'checkpoint.pth')
		if os.path.exists(checkpoint_file):
			checkpoint = torch.load(checkpoint_file)
			self.model.load_state_dict(checkpoint['model_state_dict'])
			self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
			chunk_idx = checkpoint['chunk_idx'] + 1
			logger.info(f"Resuming from chunk index {chunk_idx}.")
		else:
			logger.error("No checkpoint found. Please provide a model and checkpoint.")

	def train_model_on(self, data_file_path, chunk_size=int(1e6)):
		"""
		Chunk size is the number of instances, after which, the model will be saved, and
		:param data_file_path:
		:param chunk_size:
		:return:
		"""
		logger.info(f"Started training on {data_file_path}")
		if data_file_path[-4:] == ".txt":
			encoded_sequences, encoded_targets = create_sequences_from_book(
				VOCAB=self.VOCAB,
				text_file_path=data_file_path,
				context_window_length=self.context_window
			)
			logger.info("Obtained sequences")
			num_sequences = len(encoded_sequences)
			for chunk_id in range(0, num_sequences, chunk_size):
				logger.info(f"Training for chunk id {chunk_id}")
				self.encoded_sequences = encoded_sequences[chunk_id: chunk_id + chunk_size]
				self.encoded_targets = encoded_targets[chunk_id: chunk_id + chunk_size]
				model, optimizer = self.train()
				Train.quick_save_model(model, optimizer, chunk_id, data_file_path, QUICK_SAVE_PATH)
			self.plot_loss()
			Train.quick_save_model(model, optimizer, -1, data_file_path, MODEL_FILE_PATH)
			logger.info(f"Successfully trained on {data_file_path}")
