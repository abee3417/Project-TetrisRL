import os
import shutil
from random import random, randint, sample
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from Qnet_simple import SimpleQNetwork
from Qnet_deep import DeepQNetwork
from Qnet_double import DoubleQNetwork

from tetris import Tetris

# 파라메터 조정
SEED = 42
USE_CUDA = True
RENDER = True  # 트레이닝 과정 테트리스 출력 여부
WIDTH = 10
HEIGHT = 20
BLOCK_SIZE = 30
GAMMA = 0.99
INITIAL_EPSILON = 1
FINAL_EPSILON = 1e-3
DECAY_EPOCHS = 400
EPOCHS = 500
BATCH_SIZE = 512
MAX_STEPS = 500
LOG_PATH = "tensorboard"
SAVE_INTERVAL = 1000
DROP_SPEED = 0  # 블록이 떨어지는 속도 (초), 0 설정시 비활성화

class Agent:
    def __init__(self, device):
        self.epsilon = INITIAL_EPSILON
        self.initial_epsilon, self.final_epsilon = INITIAL_EPSILON, FINAL_EPSILON
        self.epsilon_decay_step = DECAY_EPOCHS

        self.batch_size = BATCH_SIZE
        self.update_target_rate = 10000

        self.replay_memory = deque(maxlen=30000)

        # 모델 생성
        self.model = DeepQNetwork()  # 모델 여기에다 수정, SimpleQNetwork, DeepQNetwork, DoubleQNetwork
        self.model_name = self.model.get_name()
        self.main_network = self.model.to(device)
        self.target_q_network = self.model.to(device)
        self.target_q_network.eval()
        self.update_target()

    # epsilon 값을 계산하는 함수
    def calc_epsilon(self, epoch):
        epsilon = FINAL_EPSILON + (max(DECAY_EPOCHS - epoch, 0) * (
                INITIAL_EPSILON - FINAL_EPSILON) / DECAY_EPOCHS)
        return epsilon

    # 타겟 네트워크를 업데이트하는 함수
    def update_target(self):
        self.target_q_network.load_state_dict(self.main_network.state_dict())

    # 미니배치를 얻는 함수
    def get_minibatch(self):
        batch = sample(self.replay_memory, min(len(self.replay_memory), BATCH_SIZE))
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.stack(tuple(state for state in state_batch))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.stack(tuple(state for state in next_state_batch))
        return state_batch, reward_batch, next_state_batch, done_batch


def train():
    # 시드 및 장치 설정
    torch.manual_seed(SEED)
    device = 'cuda' if USE_CUDA and torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.manual_seed_all(SEED)

    # 텐서보드 로깅 설정
    if os.path.isdir(LOG_PATH):
        shutil.rmtree(LOG_PATH)
    os.makedirs(LOG_PATH)
    writer = SummaryWriter(LOG_PATH)

    # 테트리스 환경 및 에이전트 초기화
    env = Tetris(width=WIDTH, height=HEIGHT, block_size=BLOCK_SIZE, drop_speed=DROP_SPEED, render=RENDER)
    agent = Agent(device)

    optimizer = torch.optim.Adam(agent.main_network.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    state = env.reset().to(device)

    epoch = 0
    max_lines = 0  # 단일 정수로 초기화
    while epoch < EPOCHS:
        step_count = 0  # 행동 수 초기화
        while True:
            next_steps = env.get_next_states()

            # 탐험 또는 이용 결정
            epsilon = agent.calc_epsilon(epoch)
            random_action = random() <= epsilon

            next_actions, next_states = zip(*next_steps.items())
            next_states = torch.stack(next_states).to(device)

            agent.main_network.eval()
            with torch.no_grad():
                predictions = agent.main_network(next_states)[:, 0]
            agent.main_network.train()
            if random_action:
                index = randint(0, len(next_steps) - 1)
            else:
                index = torch.argmax(predictions).item()

            next_state = next_states[index, :]
            action = next_actions[index]
            reward, done = env.step(action, render=RENDER)  # 화면에 출력

            agent.replay_memory.append([state, reward, next_state, done])

            step_count += 1  # 행동 수 증가

            if done or step_count >= MAX_STEPS:  # 종료 조건 확인
                final_score = env.score
                final_num_pieces = env.tetrominoes
                final_cleared_lines = env.cleared_lines
                state = env.reset().to(device)
                break
            else:
                state = next_state

        if epoch % agent.update_target_rate == 0:
            agent.update_target()

        epoch += 1
        state_batch, reward_batch, next_state_batch, done_batch = agent.get_minibatch()
        state_batch, reward_batch, next_state_batch = state_batch.to(device), reward_batch.to(device), next_state_batch.to(device)

        q_values = agent.main_network(state_batch)

        agent.main_network.eval()
        with torch.no_grad():
            next_prediction_batch = agent.target_q_network(next_state_batch)

        agent.main_network.train()
        y_batch = torch.cat(
            tuple(reward if done else reward + GAMMA * prediction
                  for reward, done, prediction in zip(reward_batch, done_batch, next_prediction_batch))
        )[:, None]

        optimizer.zero_grad()
        loss = criterion(q_values, y_batch)

        if final_cleared_lines > max_lines:
            max_lines = final_cleared_lines

        loss.backward()
        optimizer.step()

        print("Episode: {}/{}, Score: {}, Clear lines: {}, Blocks(Actions): {}".format(
            epoch,
            EPOCHS,
            final_score,
            final_cleared_lines,
            final_num_pieces,
            ))

        # 텐서보드에 기록
        writer.add_scalar('Train/Score', final_score, epoch)
        writer.add_scalar('Train/Tetrominoes', final_num_pieces, epoch)
        writer.add_scalar('Train/Cleared_lines', final_cleared_lines, epoch)

        # 모델 저장
        if epoch > 0 and epoch % SAVE_INTERVAL == 0:
            torch.save(agent.main_network, "./{}_tetris_model_{}".format(agent.model_name, epoch))

    torch.save(agent.main_network, "./{}_tetris_model".format(agent.model_name))
    writer.close()

if __name__ == "__main__":
    train()