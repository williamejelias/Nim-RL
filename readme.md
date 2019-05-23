# Nim Reinforcement Learning Agents

## Introduction

This repository contains the code that was developed as part of my 3rd year Computer Science research thesis titled "Analysis of Deep Reinforcement Learning on Nim". Nim is a two-player combinatorial game with a trivial solution that allows us to know the optimal moves available at each game state with a simple function. The overall goal of the project was to gain a better understanding of why certain reinforcement learning algorithms are successful/unsuccessful by analysing how they learn optimal transitions and partition states into a win/lose classification.

Implementation with Keras and the TensorFlow backend of various Reinforcement Learning algorithms for the game of Nim.

Agent Algorithms:
* Tabular Q-Learning Algorithm
* Deep Q-Network
* Deep SARSA
* Double-Deep Q-Network

Also implemented is a classifier network along with a complete and partial dataset of environment transitions and the classified optimal moves for those game states as labels. This allows a comparison between Reinforcement and Supervised Learning.

## Usage

To use the trainer program, you must have the Nim Environment installed - refer to my other repository `gym-nim` for installation instructions.

```bash
cd Nim-RL
python3 Trainer.py
```

To tweak experiments, change values within the Trainer program - within this program you can train two agents against each other and evaluate both at regular intervals against a third agent. A self play setting is also given, allowing an agent to train against itself and thus learn both sides of the interactions with the environment