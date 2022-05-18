# Generative-Music-from-Learned-Emotional-Cues - Proposing a New Architecture

## Abstract
The Oxford dictionary defines ‘music’ as sounds that are arranged in a way that is pleasant or exciting to listen to. Having a good taste of music or the ability to play a music instrument is a skill. Unfortunately, not everyone possesses  it. This project is an attempt to allow one to generate music of different emotions using WaveNet architecture and Resnet architecture. WaveNet and ResNet are one of the most common architectures for generating and classifying music. In this project, we designed an architecture which modified the WaveNet architecture to generate music of different genres and then integrated a ResNet architecture which classified the different music generated. As part of future steps, we plan to go one step further and classify music further based on emotion content in the music.

## Methodology

In this project, we took leverage of the existing WaveNet architecture and modified the architecture in such a way that it not only produced music but also gave a sense of emotional content of the music generated.

This music generated was then classified further based on its genres and emotion.

## Dataset used

We have used GTZAN dataset for Genre Classification, Chaconne compilation for WaveNet generation and GEMS Scale for Emotion Classification.

## Features
- Automatic creation of a dataset (training and validation/test set) from all sound files (.wav, .aiff, .mp3) in a directory
- Efficient multithreaded data loading
- Fast generation
- Genre Classification
- Emotion Classification
- Classifiers can classify any .wav, .mp3 file for its genre and emotion content. Classifiers can work alone as well.

## How to Run

There are two ways of running this project
1. Use Existing Trained Models with .pth file 
2. Generate from scratch.

1. Use Existing Trained Models with .pth file 

- There is already a sample generated music at  pytorch-wavenet/generated_samples/ with the name latest_generated_clip.wav. This can be used directly.

- There are two separate .pth file named model_emotion.pth and model_genre.pth. Use these files and load them on the model on classification_model.py file. 

- This can be done by running the file emotion_and_genres_generator.ipynb. It already had the relevant path inside it.

- All the outputs containing different genre and emotion audios and images are stored in ./output/

2. Generating and training from scratch New .pth file

- An altogether new clip can be generated and trained by WaveNet architecture by executing all the lines in the file /pytorch-wavenet/wavenet_demo_script.py. It already has all other required filepaths inplace.

- To train the classifier for genre, execute all the lines in the file genre_classification_training.py, it creates a new model_genre.pth.

- Similarly to train the classifier for emotions, execute all the lines in the file emotion_classification_training.py, it creates a new model_emotion.pth.

- Now run emotion_and_genres_generator.ipynb to generate different outputs which are stored in ./output

## Results

We were able to generate varied emotions and genres from a single musical piece. All the images in ./output show us how they are interrelated. With this project we were able to successfully extend the WaveNet architecture, by adding our own novelty. 

As of now these models were trained on a small dataset. When we are able to find more suitable dataset, we would try improving the performance of this modified architecture.



