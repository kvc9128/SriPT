import re
import json
import logging
import numpy as np
from Code.Utilities.Transcoder import encode_raw_text

logger = logging.getLogger(__name__)

def shuffle_sequences(encoded_sequences, encoded_targets):
    encoded_sequences = np.array(encoded_sequences)
    encoded_targets = np.array(encoded_targets)
    indices = np.arange(len(encoded_sequences))
    np.random.shuffle(indices)
    return encoded_sequences[indices], encoded_targets[indices]

def read_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        logger.warning(f"{file_path} not found, returning empty list.")
        return np.array([]), np.array([])

def create_padded_sequence(tokens, context_window_length):
    """
    tokens = tokenized_context + [sep_token] + tokenized_question + tokenized_answer_parts
    :param tokens:
    :param context_window_length:
    :return:
    """
    pad_token = "PAD"
    padding_length = context_window_length - len(tokens)
    return [pad_token] * padding_length + tokens

def generate_padded_sequence_with_context(context, question, vocab, context_window_length, answer_parts):
    """
    Given some context, and question it will generate an encoded padded sequence for you
    :param context: A string containing some information
    :param question: A question based on the context
    :param vocab:
    :param context_window_length:
    :param answer_parts:
    :return:
    """
    sep_token = "SEP"

    tokenized_context = vocab.tokenize_sentence(context)
    tokenized_question = vocab.tokenize_sentence(question)

    if answer_parts:
        tokenized_answer_parts = vocab.tokenize_sentence(answer_parts)
    else:
        tokenized_answer_parts = []

    combined_tokens = tokenized_context + [sep_token] + tokenized_question + tokenized_answer_parts
    padded_sequence = create_padded_sequence(combined_tokens, context_window_length)

    return padded_sequence

def create_sequences_from_book(vocab, text_file_path, context_window_length):
    try:
        with open(text_file_path, encoding='utf-8') as f:
            text = f.read()
            eos_token = "EOS"
            sentence_with_eos = re.sub(r'\.', f' {eos_token}', text)
            normalized_words = vocab.tokenize_sentence(sentence_with_eos)

            # TODO: DEBUGGING PURPOSES ONLY
            # normalized_words = normalized_words[:1000]

            logger.info(f"Found ~{len(normalized_words)} words in book")

            encoded_sequences, encoded_targets = [], []
            for i in range(0, len(normalized_words) - 1 - context_window_length, 256):
                question_tokens = normalized_words[i: i + context_window_length]
                target_tokens = normalized_words[i + 1: i + context_window_length + 1]
                for t in range(3, len(question_tokens)):
                    sequence = generate_padded_sequence_with_context(
                        context="",
                        question=" ".join(question_tokens[:t + 1]),
                        vocab=vocab,
                        context_window_length=context_window_length,
                        answer_parts=""
                    )
                    target = target_tokens[t]

                    encoded_sequence = encode_raw_text(sequence, vocab, context_window_length, inference=False)
                    encoded_target = np.array([vocab.word2index(target)])

                    encoded_sequences.append(encoded_sequence)
                    encoded_targets.append(encoded_target)

            return shuffle_sequences(encoded_sequences, encoded_targets)

    except FileNotFoundError:
        logger.warning("File not found, returning empty list.")
        return np.array([]), np.array([])

def create_sequences_from_json(vocab, json_file_path, context_window_length):
    if json_file_path == "../Datasets/QA/common_sense_q_a.json":
        return create_sequences_from_common_sense(vocab, context_window_length)
    elif json_file_path == "../Datasets/QA/squad.json":
        return create_sequences_from_squad(vocab, context_window_length)
    elif json_file_path == "../Datasets/QA/trivia.json":
        return create_sequences_from_trivia(vocab, context_window_length)

def create_sequences_from_common_sense(vocab, context_window_length):
    encoded_sequences, encoded_targets = [], []
    try:
        file_path = "../Datasets/QA/common_sense_q_a.json"
        with open(file_path, 'r') as file:
            for line in file:
                # Parse the JSON object in each line
                json_obj = json.loads(line)

                question_text = json_obj['question']['stem']
                correct_label = json_obj['answerKey']
                # Extract words from the correct choice
                choice_text = ""
                for choice in json_obj['question']['choices']:
                    if choice['label'] == correct_label:
                        choice_text += choice['text'] + " EOS"
                        break  # Exit the loop once the correct choice is found

                tokenized_answer = vocab.tokenize_sentence(choice_text)
                for t in range(len(tokenized_answer)):
                    part_answer = " ".join(tokenized_answer[:t])
                    sequence = generate_padded_sequence_with_context(
                        context="",
                        question=question_text,
                        vocab=vocab,
                        context_window_length=context_window_length,
                        answer_parts=part_answer
                    )
                    target = tokenized_answer[t]
                    encoded_sequence = encode_raw_text(sequence, vocab, context_window_length,
                                                       inference=False)
                    encoded_target = np.array([vocab.word2index(target)])

                    encoded_sequences.append(encoded_sequence)
                    encoded_targets.append(encoded_target)

        logger.info(
            msg=f"Generated and shuffled {len(encoded_sequences)} sequences for common sense dataset")
        return shuffle_sequences(encoded_sequences, encoded_targets)

    except FileNotFoundError:
        logger.warning(msg="CommonSenseQA not found, returning empty list.")
        return np.array([]), np.array([])


def create_sequences_from_squad(vocab, context_window_length):
    encoded_sequences, encoded_targets = [], []
    data = read_json_file("../Datasets/QA/squad.json")
    for item in data['data']:
        for paragraph in item['paragraphs']:
            context = paragraph['context']

            for qa in paragraph['qas']:
                question = qa['question']
                answer_text = qa['answers'][0]['text'] if qa['answers'] else "NO ANSWER"
                answer_text = answer_text + " EOS"  # Append EOS to the end of the answer

                tokenized_answer = vocab.tokenize_sentence(answer_text)

                for t in range(len(tokenized_answer)):
                    part_answer = " ".join(tokenized_answer[:t])
                    sequence = generate_padded_sequence_with_context(
                        context=context,
                        question=question,
                        vocab=vocab,
                        context_window_length=context_window_length,
                        answer_parts=part_answer
                    )
                    target = tokenized_answer[t]
                    encoded_sequence = encode_raw_text(sequence, vocab, context_window_length, inference=False)
                    encoded_target = np.array([vocab.word2index(target)])

                    encoded_sequences.append(encoded_sequence)
                    encoded_targets.append(encoded_target)

    logger.info(
        msg=f"Generated and shuffled {len(encoded_sequences)} sequences for SQUAD dataset")
    return shuffle_sequences(encoded_sequences, encoded_targets)
    

def create_sequences_from_trivia(vocab, context_window_length):
    encoded_sequences, encoded_targets = [], []
    data = read_json_file("../Datasets/QA/trivia.json")

    for entry in data['Data']:
        question = entry['Question']
        normalized_entity_name = entry['Answer']['Value']
        context = entry['SearchResults'][0]['Description'] if entry['SearchResults'] else ""
        normalized_entity_name = normalized_entity_name + " EOS"  # Append EOS to the end of the answer

        tokenized_answer = vocab.tokenize_sentence(normalized_entity_name)

        for t in range(len(tokenized_answer)):
            part_answer = " ".join(tokenized_answer[:t])
            sequence = generate_padded_sequence_with_context(
                context=context,
                question=question,
                vocab=vocab,
                context_window_length=context_window_length,
                answer_parts=part_answer
            )
            target = tokenized_answer[t]
            encoded_sequence = encode_raw_text(sequence, vocab, context_window_length, inference=False)
            encoded_target = np.array([vocab.word2index(target)])

            encoded_sequences.append(encoded_sequence)
            encoded_targets.append(encoded_target)
    logger.info(
        msg=f"Generated and shuffled {len(encoded_sequences)} sequences for Trivia dataset")
    return shuffle_sequences(encoded_sequences, encoded_targets)
