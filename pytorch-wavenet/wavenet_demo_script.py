# %% [markdown]
# This is notebook gives a quick overview of this WaveNet implementation, i.e. creating the model and the data set, training the model and generating samples from it.

# %%
import torch
from wavenet_model import *
from audio_data import WavenetDataset
from wavenet_training import *
from model_logging import *
import os

# %% [markdown]
# ## Model
# This is an implementation of WaveNet as it was described in the original paper (https://arxiv.org/abs/1609.03499). Each layer looks like this:
# 
# ```
#             |----------------------------------------|      *residual*
#             |                                        |
#             |    |-- conv -- tanh --|                |
#  -> dilate -|----|                  * ----|-- 1x1 -- + -->  *input*
#                  |-- conv -- sigm --|     |
#                                          1x1
#                                           |
#  ---------------------------------------> + ------------->  *skip*
# ```
# 
# Each layer dilates the input by a factor of two. After each block the dilation is reset and start from one. You can define the number of layers in each block (``layers``) and the number of blocks (``blocks``). The blocks are followed by two 1x1 convolutions and a softmax output function.
# Because of the dilation operation, the independent output for multiple successive samples can be calculated efficiently. With ``output_length``, you can define the number these outputs. Empirically, it seems that a large number of skip channels is required.

# %%
# initialize cuda option
dtype = torch.FloatTensor # data type
ltype = torch.LongTensor # label type

use_cuda = torch.cuda.is_available()
# use_cuda = False
if use_cuda:
    print('use gpu')
    dtype = torch.cuda.FloatTensor
    ltype = torch.cuda.LongTensor

# %%
model = WaveNetModel(layers=10,
                     blocks=3,
                     dilation_channels=32,
                     residual_channels=32,
                     skip_channels=1024,
                     end_channels=512, 
                     output_length=16,
                     dtype=dtype,
                     bias=True)
if use_cuda:
    model.cuda()
# model = load_latest_model_from('snapshots', use_cuda=use_cuda)

print('model: ', model)
print('receptive field: ', model.receptive_field)
print('parameter count: ', model.parameter_count())

# %% [markdown]
# ## Data Set
# To create the data set, you have to specify a path to a data set file. If this file already exists it will be used, if not it will be generated. If you want to generate the data set file (a ``.npz`` file), you have to specify the directory (``file_location``) in which all the audio files you want to use are located. The attribute ``target_length`` specifies the number of successive samples are used as a target and corresponds to the output length of the model. The ``item_length`` defines the number of samples in each item of the dataset and should always be ``model.receptive_field + model.output_length - 1``.
# 
# ```
#           |----receptive_field----|
#                                 |--output_length--|
# example:  | | | | | | | | | | | | | | | | | | | | |
# target:                           | | | | | | | | | |  
# ```
# To create a test set, you should define a ``test_stride``. Then each ``test_stride``th item will be assigned to the test set.

# %%
data = WavenetDataset(dataset_file='train_samples/bach_chaconne/dataset.npz',
                      item_length=model.receptive_field + model.output_length - 1,
                      target_length=model.output_length,
                      file_location='train_samples/bach_chaconne',
                      test_stride=500, 
                      dtype=dtype)
print('the dataset has ' + str(len(data)) + ' items')

# %% [markdown]
# ## Training and Logging
# This implementation supports logging with TensorBoard (you need to have TensorFlow installed). You can even generate audio samples from the current snapshot of the model during training. This will happen in a background thread on the cpu, so it will not interfere with the actual training but will be rather slow. If you don't have TensorFlow, you can use the standard logger that will print out to the console.
# The trainer uses Adam as default optimizer.

# %%
def generate_and_log_samples(step):
    sample_length=32000
    gen_model = load_latest_model_from('snapshots', use_cuda=False)
    print("start generating...")
    samples = generate_audio(gen_model,
                             length=sample_length,
                             temperatures=[0.5])
    tf_samples = tf.convert_to_tensor(samples, dtype=dtype)
    logger.audio_summary('temperature_0.5', tf_samples, step, sr=16000)

    samples = generate_audio(gen_model,
                             length=sample_length,
                             temperatures=[1.])
    tf_samples = tf.convert_to_tensor(samples, dtype=dtype)
    logger.audio_summary('temperature_1.0', tf_samples, step, sr=16000)
    print("audio clips generated")


# logger = TensorboardLogger(log_interval=200,
#                            validation_interval=400,
#                            generate_interval=1000,
#                            generate_function=generate_and_log_samples,
#                            log_dir="logs/chaconne_model")

logger = Logger(log_interval=200,
                validation_interval=400,
                generate_interval=1000)

# %%
trainer = WavenetTrainer(model=model,
                         dataset=data,
                         lr=0.001,
                         snapshot_path='snapshots',
                         snapshot_name='chaconne_model',
                         snapshot_interval=1000,
                         logger=logger,
                         dtype=dtype,
                         ltype=ltype)
                         
print('start training...')
trainer.train(batch_size=16,
              epochs=10)

# %% [markdown]
# ## Generating
# This model has the Fast Wavenet Generation Algorithm (https://arxiv.org/abs/1611.09482) implemented. This might run faster on the cpu. You can give some starting data (of at least the length of receptive field) or let the model generate from zero. In my experience, a temperature between 0.5 and 1.0 yields the best results, but this may depend on the data set.

# %%
start_data = data[250000][0] # use start data from the data set
start_data = torch.max(start_data, 0)[1] # convert one hot vectors to integers

def prog_callback(step, total_steps):
    print(str(100 * step // total_steps) + "% generated")

generated = model.generate_fast(num_samples=160000,
                                 first_samples=start_data,
                                 progress_callback=prog_callback,
                                 progress_interval=1000,
                                 temperature=1.0,
                                 regularize=0.)

# %%
import IPython.display as ipd

ipd.Audio(generated, rate=16000)


