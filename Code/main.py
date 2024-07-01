import json
import os
import torch
import logging
import vocab_manager

import numpy as np
import torch.nn as nn

from Code.Train.text_file_parser import generate_padded_sequence_with_context
from Model.SriPT import SriPT
from Train.Train import Train
from model_hyperparameters import EPOCHS
from model_hyperparameters import BATCH_SIZE
from model_hyperparameters import DROPOUT_RATE
from model_hyperparameters import CONTEXT_WINDOW
from model_hyperparameters import EMBEDDING_DIMENSION
from model_hyperparameters import NUM_ATTENTION_HEADS
from model_hyperparameters import NUM_DECODER_LAYERS
from Code.Utilities.Transcoder import encode_raw_text


logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
datasets = [
    "../Datasets/QA/common_sense_q_a.json",
    # "../Datasets/QA/trivia.json",
    # "../Datasets/QA/squad.json",
    # "../Datasets/Books/REUTERS_NEWS.txt",
    # "../Datasets/QA/squad.json",
]
SAVED_FOLDER = "../TRAINED_MODELS/reuters_base.pt"
SOFTMAX = nn.Softmax(dim=-1)

def load_model(model):
    try:
        if os.path.exists(SAVED_FOLDER):
            model_path = os.path.join(SAVED_FOLDER, 'reuters_base.pt')
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

def evaluate_on_common_sense(model, VOCAB):
    eval_path = "../Datasets/Evaluation/common_sense_qa_dev.json"
    with open(eval_path, 'r') as file:
        data = json.load(file)
        sample_questions = np.random.choice(data, size=5, replace=False)
    logger.info("=" * 50)
    logger.info(" Common Sense QA Dev evaluation")
    logger.info("=" * 50)
    model.eval()
    with torch.no_grad():
        for item in sample_questions:
            question = item['Question']
            correct_answer = item['Answer']['Value']
            context = ""
            generated_answer = ""

            for _ in range(25):  # 25 is an arbitrary limit for token generation
                sequence = generate_padded_sequence_with_context(
                    context=context,
                    question=question,
                    vocab=VOCAB,
                    context_window_length=CONTEXT_WINDOW,
                    answer_parts=generated_answer
                )

                encoded_sequence = torch.tensor(encode_raw_text(sequence, VOCAB, CONTEXT_WINDOW),
                                                dtype=torch.long, device=DEVICE).unsqueeze(0)
                mask_tensor = torch.ones_like(encoded_sequence, dtype=torch.long, device=DEVICE)
                mask_tensor[encoded_sequence == VOCAB.word2index("PAD")] = 0

                output = model(encoded_sequence, mask_tensor)
                output = SOFTMAX(output)
                predicted_token_id = torch.multinomial(output, 1).item()
                predicted_token = VOCAB.index2word(predicted_token_id)

                if predicted_token in ["EOS", "eos"]:
                    break
                else:
                    generated_answer += " " + predicted_token

            logger.info(f"Context: {context}")
            logger.info(f"Question: {question}")
            logger.info(f"Correct Answer: {correct_answer}")
            logger.info(f"Predicted Answer: {generated_answer.strip()}")
            logger.info("-" * 50)

def evaluate_on_squad_dev(model, vocab):
    eval_path = "../Datasets/Evaluation/squad_dev.json"
    with open(eval_path, 'r') as file:
        data = json.load(file)

    model.eval()
    total, correct = 0, 0
    results = []
    with torch.no_grad():
        for item in data['data']:
            for paragraph in item['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    question = qa['question']
                    answers = qa['answers']
                    if not answers:
                        continue
                    correct_answer = answers[0]['text']
                    generated_answer = ""

                    for _ in range(5):  # 25 is an arbitrary limit for token generation
                        sequence = generate_padded_sequence_with_context(
                            context=context,
                            question=question,
                            vocab=vocab,
                            context_window_length=CONTEXT_WINDOW,
                            answer_parts=generated_answer
                        )

                        encoded_sequence = torch.tensor(encode_raw_text(sequence, vocab, CONTEXT_WINDOW),
                                                        dtype=torch.long, device=DEVICE).unsqueeze(0)
                        mask_tensor = torch.ones_like(encoded_sequence, dtype=torch.long, device=DEVICE)
                        mask_tensor[encoded_sequence == vocab.word2index("PAD")] = 0

                        output = model(encoded_sequence, mask_tensor)
                        output = SOFTMAX(output[:, -1, :])  # Get the output for the last token in the sequence
                        predicted_token_id = torch.multinomial(output, 1).item()
                        predicted_token = vocab.index2word(predicted_token_id)

                        if predicted_token in ["EOS", "eos"]:
                            break
                        else:
                            generated_answer += " " + predicted_token

                    generated_answer = generated_answer.strip()
                    logger.info(f"Context: {context}")
                    logger.info(f"Question: {question}")
                    logger.info(f"Correct Answer: {correct_answer}")
                    logger.info(f"Predicted Answer: {generated_answer}")
                    logger.info("-" * 50)

                    if correct_answer in generated_answer or generated_answer in correct_answer:
                        correct += 1
                    total += 1
                    results.append((context, question, correct_answer, generated_answer))

    accuracy = correct / total if total > 0 else 0
    logger.info(f"SQuAD Dev Set Evaluation - Accuracy: {accuracy * 100:.2f}%")

def main():
    VOCAB = vocab_manager.load_vocab()
    VOCAB_SIZE = VOCAB.num_words()
    logger.info(msg=f"Running on {DEVICE}")
    MODEL = SriPT(
        number_of_tokens=VOCAB_SIZE,
        max_sequence_length=CONTEXT_WINDOW,
        embedding_dimension=EMBEDDING_DIMENSION,
        number_of_layers=NUM_DECODER_LAYERS,
        number_of_heads=NUM_ATTENTION_HEADS,
        dropout_rate=DROPOUT_RATE
    )
    MODEL.to(DEVICE)
    OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=0.0001)
    LOSS_FN = nn.CrossEntropyLoss()

    trainer = Train(
        model=MODEL,
        batch_size=BATCH_SIZE,
        num_epochs=EPOCHS,
        loss_fn=LOSS_FN,
        optimizer=OPTIMIZER,
        context_window=CONTEXT_WINDOW
    )

    total_params = sum(
        param.numel() for param in MODEL.parameters()
    )
    logger.info(f"Model has {total_params} parameters.")

    # TODO: Uncomment depending on if you are retraining or building form scratch
    # MODEL = load_model(MODEL)
    for dataset in datasets:
        trainer.train_model_on(dataset)
        MODEL = load_model(MODEL)
        trainer.update_model_and_optimizer(MODEL, OPTIMIZER)
        evaluate_on_common_sense(MODEL, VOCAB)
    # evaluate finally on squad dev
    evaluate_on_squad_dev(MODEL, VOCAB)


main()
