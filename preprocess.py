import os
import pickle
import copy


CODES = {'<PAD>': 0, '<EOS>': 1, '<UNK>': 2, '<GO>': 3}


def load_data(path):
    """
    Load Dataset from File
    """
    input_file = os.path.join(path)
    with open(input_file, 'r', encoding='utf-8') as f:
        return f.read()


def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    """
    vocab = set(text.split())
    vocab_to_int = copy.copy(CODES)

    for v_i, v in enumerate(vocab, len(CODES)):
        vocab_to_int[v] = v_i

    int_to_vocab = {v_i: v for v, v_i in vocab_to_int.items()}

    return vocab_to_int, int_to_vocab


# Text to Word Ids

def text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int):
    """
    Convert source and target text to proper word ids
    :param source_text: String that contains all the source text.
    :param target_text: String that contains all the target text.
    :param source_vocab_to_int: Dictionary to go from the source words to an id
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: A tuple of lists (source_id_text, target_id_text)
    """
    # TODO: Implement Function
    source_id_text = []
    for source_sentence in source_text.split('\n'):
        source_id_sentence = [source_vocab_to_int[word] for word in source_sentence.split()]
        source_id_text.append(source_id_sentence)

    target_id_text = []
    for target_sentence in target_text.split('\n'):
        target_id_sentence = [target_vocab_to_int[word] for word in target_sentence.split()]
        target_id_sentence.append(target_vocab_to_int['<EOS>'])
        target_id_text.append(target_id_sentence)
    return source_id_text, target_id_text


def preprocess_and_save_data(source_path, target_path, text_to_ids):
    """
    Preprocess Text Data.  Save to to file.
    """
    # Preprocess
    print('Preprocessing...')

    source_text = load_data(source_path)
    target_text = load_data(target_path)

    source_text = source_text.lower()
    target_text = target_text.lower()

    source_vocab_to_int, source_int_to_vocab = create_lookup_tables(source_text)
    target_vocab_to_int, target_int_to_vocab = create_lookup_tables(target_text)

    source_text, target_text = text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int)

    # Save Data
    with open('preprocess.p', 'wb') as out_file:
        pickle.dump(((source_text, target_text),
                     (source_vocab_to_int, target_vocab_to_int),
                     (source_int_to_vocab, target_int_to_vocab)), out_file)
    print('All data have been preprocessed and saved to preprocess.p.')


# Get the Data

source_path = 'data/small_vocab_en'
target_path = 'data/small_vocab_fr'
source_text = load_data(source_path)
target_text = load_data(target_path)


# Preprocess all the data and save it

preprocess_and_save_data(source_path, target_path, text_to_ids)
