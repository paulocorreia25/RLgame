import random
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, InputLayer, Input, Reshape, ConvLSTM2D, LSTM, Lambda, Dropout
from tensorflow.keras.optimizers import Adam
from  tensorflow.keras.models import Model
import os
import math
import numpy as np
from rl.agents import DQNAgent, DDPGAgent
from rl.policy import BoltzmannQPolicy, GreedyQPolicy
from rl.memory import SequentialMemory
import gym

tf.compat.v1.experimental.output_all_intermediates(True)


class Game:
    def __init__(self):
        self.visualize = False
        self.symbols = {'empty': [1, 0, 0, 0, 0], 'food': [0, 1, 0, 0, 0], 'player': [0, 0, 1, 0, 0], 'wall': [0, 0, 0, 1, 0], 'mine': [0, 0, 0, 0, 1]}
        self.x_pos = 2
        self.food = 20
        self.x_dim = 30
        self.y_dim = 8
        self.moves = 0
        self.game_map = []
        self.draw()
        self.spawn_food(self.food)
        self.player_pos = (int(self.x_dim / 2), self.y_dim-2)

    def draw(self):
        temp = []
        for j in range(self.y_dim):
            x = []
            if j == 0 or j == self.y_dim - 1:
                for i in range(self.x_dim):
                    x.append(self.symbols['wall'])
            else:
                for i in range(self.x_dim):
                    if i == 0 or i == self.x_dim - 1:
                        x.append(self.symbols['wall'])
                    else:
                        x.append(self.symbols['empty'])
            temp.append(x)
        self.game_map = temp
        self.game_map[self.y_dim-2][self.x_pos] = self.symbols['player']

    def spawn_food(self, amount):
        n = amount
        while n != 0:
            x_rand, y_rand = random.randint(1, self.x_dim - 2), random.randint(1, self.y_dim - 2)
            if self.game_map[y_rand][x_rand] == self.symbols['empty']:
                self.game_map[y_rand][x_rand] = self.symbols['food']
                n -= 1

    def get_pos(self, item):
        for y in range(self.y_dim):
            for x in range(self.x_dim):
                if self.game_map[y][x] == item:
                    yield x, y

    def spawn(self):
        num = random.randint(1, 15)
        if num == 1:
            return self.symbols['food']
        if num == 4 or num == 5:
            return self.symbols['mine']
        return self.symbols['empty']

    def move_map(self):
        temp = []
        for i, x in enumerate(self.game_map):
            if i != 0 and i != self.y_dim-1:
                player = list(self.get_pos(self.symbols['player']))[0]
                if i != player[1]:
                    food = self.spawn()
                    temp.append([self.symbols['wall'], *x[2:-1], food, self.symbols['wall']])
                else:
                    food = self.spawn()
                    temp2 = [self.symbols['wall'], *x[2:-1], food, self.symbols['wall']]
                    temp2[temp2.index(self.symbols['player'])] = self.symbols['empty']
                    temp2[self.x_pos] = self.symbols['player']
                    temp.append(temp2)
            else:
                temp.append(x)
        self.game_map = temp

    def reset(self):
        self.moves = 0
        self.draw()
        self.spawn_food(self.food)
        return np.array(self.game_map, dtype='float32')

    def get_map(self):
        transposed_map = np.transpose(self.game_map, (1, 0, 2))
        final_map = np.expand_dims(transposed_map, axis=1)
        # food = []
        # for y in range(self.y_dim):
        #     xs = []
        #     for x in range(self.x_dim):
        #         xs.append(self.game_map[y][x])
        #     food.append(xs)
        # food = np.array(food, dtype='float32')
        return transposed_map

    def step(self, action):
        if self.visualize:
            self.render()
            time.sleep(.5)
        acts = [-1, 0, 1]
        y = acts[action] if self.player_pos[1] + acts[action] != 0 and self.player_pos[1] + acts[action] != self.y_dim - 1 else 0
        done = True if self.moves == 500 else False
        reward = 0
        if self.game_map[self.player_pos[1] + y][self.x_pos+1] == self.symbols['food']:
            reward = 2*(self.moves+1)
        if self.game_map[self.player_pos[1] + y][self.x_pos+1] == self.symbols['mine']:
            done = True
        # if self.game_map[self.player_pos[1] + y][self.x_pos+1] == self.symbols['empty']:
        #     reward = -.5
        self.move(0, y)
        self.moves += 1
        self.move_map()
        return np.array(self.game_map, dtype='float32'), reward, done, {}

    def move(self, x, y):
        if self.player_pos[1] + y != self.y_dim - 1 and self.player_pos[1] + y != 0:
            self.game_map[self.player_pos[1]][self.player_pos[0]] = self.symbols['empty']
            self.game_map[self.player_pos[1] + y][self.x_pos] = self.symbols['player']
            self.player_pos = (self.x_pos, self.player_pos[1] + y)

    def render(self):
        translate = {str(self.symbols['empty']): ' ', str(self.symbols['food']): 'O', str(self.symbols['player']): 'X', str(self.symbols['wall']): '#', str(self.symbols['mine']): '!'}
        os.system('cls')
        for n in self.game_map:
            k = ''
            for m in n:
                k += translate[str(m)]
            print(k)
        print(self.moves)

    def run(self):
        while True:
            self.move_map()
            time.sleep(1)


game = Game()


def build_model(actions):
    shape = np.array(game.game_map, dtype='float32').shape
    print(shape)
    model = Sequential()
    model.add(Reshape(shape, input_shape=(1, *shape)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((1, 3)))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions, activation='softmax'))
    return model


def build_agent(model, actions):
    policy = GreedyQPolicy()
    memory = SequentialMemory(limit=1000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, nb_actions=actions, nb_steps_warmup=100, target_model_update=1e-2)
    return dqn


model = build_model(3)
dqn = build_agent(model, 3)
dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])
dqn.fit(game, nb_steps=20000, visualize=False, verbose=1)

game.visualize = True

scores = dqn.test(game, nb_episodes=15, visualize=False)
print(np.mean(scores.history['episode_reward']))
