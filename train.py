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

# 파라미터 설정
SEED = 42
USE_CUDA = True
RENDER = False  # 트레이닝 과정 테트리스 출력 여부
WIDTH = 10
HEIGHT = 20
BLOCK_SIZE = 30
GAMMA = 0.99
INITIAL_EPSILON = 1
FINAL_EPSILON = 1e-3
DECAY_EPOCHS = 1000
EPOCHS = 2000
BATCH_SIZE = 512
MAX_STEPS = 1000
LOG_PATH = "tensorboard"
SAVE_INTERVAL = 500
DROP_SPEED = 0  # 블록이 떨어지는 속도 (초), 0 설정시 비활성화

class Agent:
    def __init__(self, model_class, device):
        self.epsilon = INITIAL_EPSILON
        self.initial_epsilon, self.final_epsilon = INITIAL_EPSILON, FINAL_EPSILON
        self.epsilon_decay_step = DECAY_EPOCHS

        self.batch_size = BATCH_SIZE
        self.update_target_rate = 10000

        self.replay_memory = deque(maxlen=30000)

        # 모델 생성
        self.model = model_class()  # 모델 클래스를 인자로 받음
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

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Train Tetris DQN agent")
    parser.add_argument("--model", type=str, choices=["simple", "deep", "double"], required=True, help="3가지 모델 중 선택하기")
    args = parser.parse_args()
    return args

def train(model_type):
    # 시드 및 장치 설정
    torch.manual_seed(SEED)
    device = 'cuda' if USE_CUDA and torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.manual_seed_all(SEED)

    # 모델 선택
    if model_type == "simple":
        model_class = SimpleQNetwork
    elif model_type == "deep":
        model_class = DeepQNetwork
    elif model_type == "double":
        model_class = DoubleQNetwork
    else:
        raise ValueError("모델이 존재하지 않습니다.")

    # 테트리스 환경 및 에이전트 초기화
    env = Tetris(width=WIDTH, height=HEIGHT, block_size=BLOCK_SIZE, drop_speed=DROP_SPEED, render=RENDER)
    agent = Agent(model_class, device)

    # 텐서보드 로깅 설정
    if os.path.isdir(LOG_PATH):
        shutil.rmtree(LOG_PATH)
    os.makedirs(LOG_PATH)

    # SummaryWriter 생성 시 filename_suffix 지정
    writer = SummaryWriter(LOG_PATH, filename_suffix="_{}".format(agent.model_name))

    optimizer = torch.optim.Adam(agent.main_network.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    state = env.reset().to(device)

    epoch = 0
    max_lines = 0  # 단일 정수로 초기화
    cumulative_rewards = []  # 누적 보상 리스트
    episode_lengths = []  # 에피소드 길이 리스트

    while epoch < EPOCHS:
        step_count = 0  # 행동 수 초기화
        total_reward = 0  # 누적 보상 초기화

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

            total_reward += reward  # 누적 보상 업데이트
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
            if model_type == "double":
                # Double Q-learning: main_network와 target_network를 모두 사용하여 업데이트
                next_main_predictions = agent.main_network(next_state_batch)
                next_target_predictions = agent.target_q_network(next_state_batch)
                max_action_indexes = torch.argmax(next_main_predictions, dim=1)
                next_q_values = next_target_predictions.gather(1, max_action_indexes.unsqueeze(1))
            else:
                # 일반 Q-learning: target_network의 최대 Q-value 사용
                next_q_values = agent.target_q_network(next_state_batch).max(1)[0].unsqueeze(1)

        agent.main_network.train()
        y_batch = reward_batch + (1 - torch.tensor(done_batch, dtype=torch.float32).to(device).unsqueeze(1)) * GAMMA * next_q_values

        # q_values와 y_batch의 크기를 맞춰줌
        q_values = q_values.squeeze(1)
        y_batch = y_batch.squeeze(1)

        optimizer.zero_grad()
        loss = criterion(q_values, y_batch)

        if final_cleared_lines > max_lines:
            max_lines = final_cleared_lines

        loss.backward()
        optimizer.step()

        print("Episode: {}/{}, Cleared lines: {}, Score: {}, Cumulative Reward: {}, Tetrominoes: {}, Episode Length: {}".format(
            epoch,
            EPOCHS,
            final_cleared_lines,
            final_score,
            total_reward,
            final_num_pieces,
            step_count
            ))

        # 누적 보상 및 에피소드 길이 기록
        cumulative_rewards.append(total_reward)
        episode_lengths.append(step_count)

        # 텐서보드에 기록
        writer.add_scalar('Train/Cleared_lines', final_cleared_lines, epoch)
        writer.add_scalar('Train/Score', final_score, epoch)
        writer.add_scalar('Train/Cumulative_Reward', total_reward, epoch)
        writer.add_scalar('Train/Tetrominoes', final_num_pieces, epoch)
        writer.add_scalar('Train/Episode_Length', step_count, epoch)

        # 500 epoch마다 모델 저장
        if epoch > 0 and epoch % SAVE_INTERVAL == 0:
            os.makedirs("./model/{}".format(agent.model_name), exist_ok=True)
            torch.save(agent.main_network, "./model/{}/tetris{}_{}".format(agent.model_name, agent.model_name, epoch))

    # 전체 에피소드의 평균 보상 기록
    average_reward = np.mean(cumulative_rewards)
    writer.add_scalar('Train/Average_Reward', average_reward, epoch)

    os.makedirs("./model/{}".format(agent.model_name), exist_ok=True)
    torch.save(agent.main_network, "./model/{}/tetris{}_final".format(agent.model_name, agent.model_name))
    writer.close()

if __name__ == "__main__":
    args = get_args()
    train(args.model)