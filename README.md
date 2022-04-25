# Generative-Music-from-Learned-Emotional-Cues - Proposing a New Architecture

## Abstract
The Oxford dictionary defines ‘music’ as sounds that are arranged in a way that is pleasant or exciting to listen to. Having a good taste of music or the ability to play a music instrument is a skill. Unfortunately, not everyone possess it. This project is an attempt to allow one to generate music of different emotions using WaveNet architecture and Resnet architecture. WaveNet and ResNet are one of the most common architectures for generating and classifying music. In this project, we designed an architecture which modified the WaveNet architecture to generate music of different genres and then integrated a ResNet architecture which classified the different music generated. As part of future steps, we plan to go one step further and classify music further based on emotion content in the music.

## Methodology

In this project, we took leverage of the existing WaveNet architecture and modified the architecture in such a way that it not only produced music but also gave a sense of emotional content of the music generated.

This music generated was then classified further based on its genres and emotion.
As of now, we have used ResNet architecture.

## Dataset used

We have used GTZAN dataset, Chaconne compilation.

## Features
- Automatic creation of a dataset (training and validation/test set) from all sound files (.wav, .aiff, .mp3) in a directory
- Efficient multithreaded data loading
- Fast generation
- Genre Classification
- Emotion Content

## How to Run
Todo: Still in progress - will update the complete flow once completed.






