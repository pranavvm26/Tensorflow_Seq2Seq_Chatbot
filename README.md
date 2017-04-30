#TensorflowSeq2SeqChatbot
Overview
============
This repo is dedicated towards building a interactive chatbot. The model utilizes the [Cornell Movie Dialogue dataset](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) to train and respond to user queries. 


Dependencies
============
* numpy
* scipy 
* six
* tensorflow (https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html)

Usage
===========

Edit the `seq2seq.ini`  to train or evaluate the model,

Set,
`mode = train`

and run,

``python execute.py``

Run the model for a few hours and then it's able to hold a fun conversation.

Similarly, to evaluate the model, edit the `seq2seq.ini` file as follows,

`mode = test`

and then run,

``python execute.py``

Output
===========

After 64000 iterations, a learning rate 0.3928, step-time 0.15 and a perplexity value of 11.58,

Model's converstaion.

> Hi .
Hi .
> Can you see me?
I ' m sorry .
> I am an owl and you can call me cow .
I ' m not going to hurt you .
> why not ?
Because I ' m not .
> come on please?
What ?
> hit me.
What ?
> this is smoking hot .
Yes .
 
