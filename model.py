import numpy as np
import tensorflow as tf
import pickle
from tensorflow.python.layers.core import Dense


# ===== Hyperparameters =====

# Number of Epochs
epochs = 10
# Batch Size
batch_size = 128
# RNN Size
rnn_size = 512
# Number of Layers
num_layers = 3
# Embedding Size
encoding_embedding_size = 256
decoding_embedding_size = 256
# Learning Rate
learning_rate = 0.001
# Dropout Keep Probability
keep_probability = 0.5
display_step = 100


# ===== Save Path =====
checkpoint_save_path = 'checkpoints/dev'


# ===== Load preprocessed data =====

def load_preprocess():
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """

    with open('preprocess.p', mode='rb') as in_file:
        return pickle.load(in_file)


# ===== Build the Neural Network =====

# Input

def model_inputs():
    """
    Create TF Placeholders for input, targets, learning rate, and lengths of source and target sequences.
    :return: Tuple (input, targets, learning rate, keep probability, target sequence length,
    max target sequence length, source sequence length)
    """

    input = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    target_seq_len = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
    max_target_len = tf.reduce_max(target_seq_len, name='max_target_len')
    source_seq_len = tf.placeholder(tf.int32, (None,), name='source_sequence_length')

    return input, targets, learning_rate, keep_prob, target_seq_len, max_target_len, source_seq_len


# Process Decoder Input

def process_decoder_input(target_data, target_vocab_to_int, batch_size):
    """
    Preprocess target data for encoding
    :param target_data: Target Placehoder
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :param batch_size: Batch Size
    :return: Preprocessed target data
    """

    with tf.name_scope('process_decoder_input'):
        rm_end = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
        preprocessed = tf.concat([tf.fill([batch_size, 1], target_vocab_to_int['<GO>']), rm_end], 1)

        return preprocessed


# Encoder

def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob,
                   source_seq_length, source_vocab_size,
                   encoding_embedding_size):
    """
    Create encoding layer
    :param rnn_inputs: Inputs for the RNN
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param keep_prob: Dropout keep probability
    :param source_seq_length: a list of the lengths of each sequence in the batch
    :param source_vocab_size: vocabulary size of source data
    :param encoding_embedding_size: embedding size of source data
    :return: tuple (RNN output, RNN state)
    """

    with tf.variable_scope('encoder'):
        with tf.name_scope('embeding'):
            embedded = tf.contrib.layers.embed_sequence(rnn_inputs, source_vocab_size, encoding_embedding_size)

        with tf.name_scope('lstm_cell'):
            def single_lstm(lstm_size):
                lstm = tf.contrib.rnn.LSTMCell(lstm_size,
                                               initializer=tf.random_uniform_initializer(-0.1, 0.1))
                dropped = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)

                return dropped

            multi_lstm = tf.contrib.rnn.MultiRNNCell([single_lstm(rnn_size) for _ in range(num_layers)])

            enc_output, enc_state = tf.nn.dynamic_rnn(multi_lstm, embedded, sequence_length=source_seq_length, dtype=tf.float32)

        return enc_output, enc_state


# Decoder - Training

def decoding_layer_train(encoder_state, dec_cell, dec_embed_input,
                         target_sequence_length, max_target_sequence_length,
                         output_layer, keep_prob):
    """
    Create a decoding layer for training
    :param encoder_state: Encoder State
    :param dec_cell: Decoder RNN Cell
    :param dec_embed_input: Decoder embedded input
    :param target_sequence_length: The lengths of each sequence in the target batch
    :param max_summary_length: The length of the longest sequence in the batch
    :param output_layer: Function to apply the output layer
    :param keep_prob: Dropout keep probability
    :return: BasicDecoderOutput containing training logits and sample_id
    """

    with tf.name_scope('train_helper'):
        train_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                         sequence_length=target_sequence_length,
                                                         time_major=False)
    with tf.name_scope('train_decoder'):
        train_decoder = tf.contrib.seq2seq.BasicDecoder(cell=dec_cell,
                                                        helper=train_helper,
                                                        initial_state=encoder_state,
                                                        output_layer=output_layer)

    with tf.name_scope('train_decoder_output'):
        train_dec_output, _ = tf.contrib.seq2seq.dynamic_decode(decoder=train_decoder,
                                                                impute_finished=True,
                                                                maximum_iterations=max_target_sequence_length)

    return train_dec_output


# Decoder - Inference

def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id,
                         end_of_sequence_id, max_target_sequence_length,
                         vocab_size, output_layer, batch_size, keep_prob):
    """
    Create a decoding layer for inference
    :param encoder_state: Encoder state
    :param dec_cell: Decoder RNN Cell
    :param dec_embeddings: Decoder embeddings
    :param start_of_sequence_id: GO ID
    :param end_of_sequence_id: EOS Id
    :param max_target_sequence_length: Maximum length of target sequences
    :param vocab_size: Size of decoder/target vocabulary
    :param decoding_scope: TenorFlow Variable Scope for decoding
    :param output_layer: Function to apply the output layer
    :param batch_size: Batch size
    :param keep_prob: Dropout keep probability
    :return: BasicDecoderOutput containing inference logits and sample_id
    """

    start_tokens = tf.tile(tf.constant([start_of_sequence_id], dtype=tf.int32), [batch_size], name='start_tokens')

    with tf.name_scope('infer_helper'):
        infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=dec_embeddings,
                                                                start_tokens=start_tokens,
                                                                end_token=end_of_sequence_id)

    with tf.name_scope('infer_decoder'):
        infer_decoder = tf.contrib.seq2seq.BasicDecoder(cell=dec_cell,
                                                        helper=infer_helper,
                                                        initial_state=encoder_state,
                                                        output_layer=output_layer)

    with tf.name_scope('infer_decoder_output'):
        infer_dec_output, _ = tf.contrib.seq2seq.dynamic_decode(decoder=infer_decoder,
                                                                impute_finished=True,
                                                                maximum_iterations=max_target_sequence_length)

    return infer_dec_output


# Decoder


def decoding_layer(dec_input, encoder_state,
                   target_sequence_length, max_target_sequence_length,
                   rnn_size,
                   num_layers, target_vocab_to_int, target_vocab_size,
                   batch_size, keep_prob, decoding_embedding_size):
    """
    Create decoding layer
    :param dec_input: Decoder input
    :param encoder_state: Encoder state
    :param target_sequence_length: The lengths of each sequence in the target batch
    :param max_target_sequence_length: Maximum length of target sequences
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :param target_vocab_size: Size of target vocabulary
    :param batch_size: The size of the batch
    :param keep_prob: Dropout keep probability
    :param decoding_embedding_size: Decoding embedding size
    :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
    """

    with tf.name_scope('decoding_layer'):

        dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]),
                                     name='dec_embeddings')
        dec_embedded = tf.nn.embedding_lookup(dec_embeddings, dec_input, name='dec_embedded')

        def single_lstm(lstm_size):
            lstm = tf.contrib.rnn.LSTMCell(lstm_size,
                                           initializer=tf.random_uniform_initializer(-0.1, 0.1))
            dropped = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
            return dropped

        dec_cell = tf.contrib.rnn.MultiRNNCell([single_lstm(rnn_size) for _ in range(num_layers)])

        output_layer = Dense(target_vocab_size,
                             kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        with tf.variable_scope('decoder'):
            train_dec_output = decoding_layer_train(encoder_state=encoder_state,
                                                    dec_cell=dec_cell,
                                                    dec_embed_input=dec_embedded,
                                                    target_sequence_length=target_sequence_length,
                                                    max_target_sequence_length=max_target_sequence_length,
                                                    output_layer=output_layer,
                                                    keep_prob=keep_prob)

        with tf.variable_scope('decoder', reuse=True):
            infer_dec_output = decoding_layer_infer(encoder_state=encoder_state,
                                                    dec_cell=dec_cell,
                                                    dec_embeddings=dec_embeddings,
                                                    start_of_sequence_id=target_vocab_to_int['<GO>'],
                                                    end_of_sequence_id=target_vocab_to_int['<EOS>'],
                                                    max_target_sequence_length=max_target_sequence_length,
                                                    vocab_size=target_vocab_size,
                                                    output_layer=output_layer,
                                                    batch_size=batch_size,
                                                    keep_prob=keep_prob)

    return train_dec_output, infer_dec_output


# Build the Neural Network

def seq2seq_model(input_data, target_data, keep_prob, batch_size,
                  source_sequence_length, target_sequence_length,
                  max_target_sentence_length,
                  source_vocab_size, target_vocab_size,
                  enc_embedding_size, dec_embedding_size,
                  rnn_size, num_layers, target_vocab_to_int):
    """
    Build the Sequence-to-Sequence part of the neural network
    :param input_data: Input placeholder
    :param target_data: Target placeholder
    :param keep_prob: Dropout keep probability placeholder
    :param batch_size: Batch Size
    :param source_sequence_length: Sequence Lengths of source sequences in the batch
    :param target_sequence_length: Sequence Lengths of target sequences in the batch
    :param source_vocab_size: Source vocabulary size
    :param target_vocab_size: Target vocabulary size
    :param enc_embedding_size: Decoder embedding size
    :param dec_embedding_size: Encoder embedding size
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
    """

    enc_output, enc_state = encoding_layer(rnn_inputs=input_data,
                                           rnn_size=rnn_size,
                                           num_layers=num_layers,
                                           keep_prob=keep_prob,
                                           source_seq_length=source_sequence_length,
                                           source_vocab_size=source_vocab_size,
                                           encoding_embedding_size=enc_embedding_size)

    dec_input = process_decoder_input(target_data, target_vocab_to_int, batch_size)

    train_dec_output, infer_dec_output = decoding_layer(dec_input=dec_input,
                                                        encoder_state=enc_state,
                                                        target_sequence_length=target_sequence_length,
                                                        max_target_sequence_length=max_target_sentence_length,
                                                        rnn_size=rnn_size,
                                                        num_layers=num_layers,
                                                        target_vocab_to_int=target_vocab_to_int,
                                                        target_vocab_size=target_vocab_size,
                                                        batch_size=batch_size,
                                                        keep_prob=keep_prob,
                                                        decoding_embedding_size=dec_embedding_size)

    return train_dec_output, infer_dec_output


# ===== Neural Network Training =====

# Pad the source and target sequences

def pad_sentence_batch(sentence_batch, pad_int):
    """
    Pad sentences with <PAD> so that each sentence of a batch has the same length
    """

    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]


# Get batches

def get_batches(sources, targets, batch_size, source_pad_int, target_pad_int):
    """
    Batch targets, sources, and the lengths of their sentences together
    """

    for batch_i in range(0, len(sources)//batch_size):
        start_i = batch_i * batch_size

        # Slice the right amount for the batch
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]

        # Pad
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))

        # Need the lengths for the _lengths parameters
        pad_targets_lengths = []
        for target in pad_targets_batch:
            pad_targets_lengths.append(len(target))

        pad_source_lengths = []
        for source in pad_sources_batch:
            pad_source_lengths.append(len(source))

        yield pad_sources_batch, pad_targets_batch, pad_source_lengths, pad_targets_lengths


# Training and validation accuracy

def get_accuracy(target, logits):
    """
    Calculate accuracy
    """

    max_seq = max(target.shape[1], logits.shape[1])
    if max_seq - target.shape[1]:
        target = np.pad(target,
                        [(0, 0), (0, max_seq - target.shape[1])],
                        'constant')
    if max_seq - logits.shape[1]:
        logits = np.pad(logits,
                        [(0, 0), (0, max_seq - logits.shape[1])],
                        'constant')

    return np.mean(np.equal(target, logits))


# Save parameters

def save_params(params):
    """
    Save parameters to file
    """

    with open('params.p', 'wb') as out_file:
        pickle.dump(params, out_file)


# Build the Graph

print('Loading preprocessed data...')

(source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = load_preprocess()
max_target_sentence_length = max([len(sentence) for sentence in source_int_text])

train_graph = tf.Graph()

with train_graph.as_default():
    input_data, targets, lr, keep_prob, target_sequence_length, max_target_sequence_length, source_sequence_length = model_inputs()

    input_shape = tf.shape(input_data)

    train_logits, inference_logits = seq2seq_model(tf.reverse(input_data, [-1]),
                                                   targets,
                                                   keep_prob,
                                                   batch_size,
                                                   source_sequence_length,
                                                   target_sequence_length,
                                                   max_target_sequence_length,
                                                   len(source_vocab_to_int),
                                                   len(target_vocab_to_int),
                                                   encoding_embedding_size,
                                                   decoding_embedding_size,
                                                   rnn_size,
                                                   num_layers,
                                                   target_vocab_to_int)

    training_logits = tf.identity(train_logits.rnn_output, name='logits')
    inference_logits = tf.identity(inference_logits.sample_id, name='predictions')

    masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')

    with tf.name_scope("optimization"):
        # Loss function
        with tf.name_scope('cost'):
            cost = tf.contrib.seq2seq.sequence_loss(
                training_logits,
                targets,
                masks)

        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)


# Split data to training and validation sets

train_source = source_int_text[batch_size:]
train_target = target_int_text[batch_size:]
valid_source = source_int_text[:batch_size]
valid_target = target_int_text[:batch_size]

(valid_sources_batch,
 valid_targets_batch,
 valid_sources_lengths,
 valid_targets_lengths) = next(get_batches(valid_source,
                                           valid_target,
                                           batch_size,
                                           source_vocab_to_int['<PAD>'],
                                           target_vocab_to_int['<PAD>']))


# Train

print('Training...')

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(epochs):
        for batch_i, (source_batch,
                      target_batch,
                      sources_lengths,
                      targets_lengths) in enumerate(get_batches(train_source,
                                                                train_target,
                                                                batch_size,
                                                                source_vocab_to_int['<PAD>'],
                                                                target_vocab_to_int['<PAD>'])):

            _, loss = sess.run([train_op, cost],
                               {input_data: source_batch,
                                targets: target_batch,
                                lr: learning_rate,
                                target_sequence_length: targets_lengths,
                                source_sequence_length: sources_lengths,
                                keep_prob: keep_probability})

            if batch_i % display_step == 0 and batch_i > 0:

                batch_train_logits = sess.run(inference_logits,
                                              {input_data: source_batch,
                                               source_sequence_length: sources_lengths,
                                               target_sequence_length: targets_lengths,
                                               keep_prob: 1.0})

                batch_valid_logits = sess.run(inference_logits,
                                              {input_data: valid_sources_batch,
                                               source_sequence_length: valid_sources_lengths,
                                               target_sequence_length: valid_targets_lengths,
                                               keep_prob: 1.0})

                train_acc = get_accuracy(target_batch, batch_train_logits)

                valid_acc = get_accuracy(valid_targets_batch, batch_valid_logits)

                print('Epoch {:>3} Batch {:>4}/{} - Train Accuracy: {:>6.4f}, Validation Accuracy: {:>6.4f}, Loss: {:>6.4f}'
                      .format(epoch_i, batch_i, len(source_int_text) // batch_size, train_acc, valid_acc, loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, checkpoint_save_path)
    print('Model Trained and Saved.')

# Save parameters

save_params(checkpoint_save_path)
