from flask import Flask, jsonify,abort,make_response,request
import os
import sys

import numpy as np
# from six.moves import range  # pylint: disable=redefined-builtin
import tensorflow as tf
import execute

import data_utils
import seq2seq_model

try:
    reload
except NameError:
    # py3k has unicode by default
    pass
else:
    reload(sys).setdefaultencoding('utf-8')

try:
    from ConfigParser import SafeConfigParser
except:
    from configparser import SafeConfigParser # In Python 3, ConfigParser has been renamed to configparser for PEP 8 compliance.

gConfig = {}

# Only allocate part of the gpu memory when predicting.
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
config = tf.ConfigProto(gpu_options=gpu_options)


app = Flask(__name__)
sess = tf.Session(config=config)

@app.route('/todo/api/v1.0/tasks/<int:task_id>', methods=['GET'])
def get_task(message):
    # task = [task for task in tasks if task['id'] == task_id]

    if len(message) == 0:
        pass
    else:
        # Decode from standard input.
        # sys.stdout.write("> ")
        # sys.stdout.flush()
        sentence = message
        # sentence = text
        while sentence:
            # Get token-ids for the input sentence.
            token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), enc_vocab)
            # Which bucket does it belong to?
            bucket_id = min([b for b in range(len(_buckets))
                             if _buckets[b][0] > len(token_ids)])
            # Get a 1-element batch to feed the sentence to the model.
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                {bucket_id: [(token_ids, [])]}, bucket_id)
            # Get output logits for the sentence.
            _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                             target_weights, bucket_id, True)
            # This is a greedy decoder - outputs are just argmaxes of output_logits.
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            # If there is an EOS symbol in outputs, cut them at that point.
            if data_utils.EOS_ID in outputs:
                outputs = outputs[:outputs.index(data_utils.EOS_ID)]
            # Print out French sentence corresponding to outputs.
            print(" ".join([tf.compat.as_str(rev_dec_vocab[output]) for output in outputs]))
            print("> ", end="")
            sys.stdout.flush()
            return_message = " ".join([tf.compat.as_str(rev_dec_vocab[output]) for output in outputs])
            return return_message



@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

@app.route('/todo/api/v1.0/tasks', methods=['POST'])
def create_task():
    if not request.json or not 'title' in request.json:
        abort(400)
    task = {
        'id': tasks[-1]['id'] + 1,
        'title': request.json['title'],
        'description': request.json.get('description', ""),
        'done': False
    }
    tasks.append(task)
    return jsonify({'task': task}), 201



# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]


# Create model and load parameters.
model = execute.create_model(sess, True)
model.batch_size = 1  # We decode one sentence at a time.

# Load vocabularies.
enc_vocab_path = os.path.join(gConfig['working_directory'],"vocab%d.enc" % gConfig['enc_vocab_size'])
dec_vocab_path = os.path.join(gConfig['working_directory'],"vocab%d.dec" % gConfig['dec_vocab_size'])

enc_vocab, _ = data_utils.initialize_vocabulary(enc_vocab_path)
_, rev_dec_vocab = data_utils.initialize_vocabulary(dec_vocab_path)

tasks = [
    {
        'id': 1,
        'title': u'Buy groceries',
        'description': u'Milk, Cheese, Pizza, Fruit, Tylenol',
        'done': False
    },
    {
        'id': 2,
        'title': u'Learn Python',
        'description': u'Need to find a good Python tutorial on the web',
        'done': False
    }
]


app.run(debug=True)
print("Flask interface opened!")




