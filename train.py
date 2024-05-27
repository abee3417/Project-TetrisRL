import argparse
import torch
import torch.nn as nn
from collections import deque
from random import random, randint, sample
from time import time
import numpy as np
from deep_q_network import DeepQNetwork  # DQN 모델을 import
from tetris import Tetris  # 테트리스 환경을 import


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_cuda", type=bool, default=True)

    parser.add_argument("--render", type=bool, default=False, help='flag - video render')
    parser.add_argument("--width", type=int, default=10, help="이미지의 공통 너비")
    parser.add_argument("--height", type=int, default=20, help="이미지의 공통 높이")
    parser.add_argument("--block_size", type=int, default=30, help="블록의 크기")

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--num_decay_epochs", type=float, default=700) # epoch의 3/4정도로 설정

    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=512, help="배치 당 이미지 수")
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--replay_memory_size", type=int, default=30000, help="재생 메모리 크기")

    parser.add_argument("--load_model", type=bool, default=False)
    parser.add_argument("--model_path", type=str, default="trained_models")

    parser.add_argument("--max_steps", type=int, default=2000, help="에피소드 당 최대 행동 수")

    args = parser.parse_args()
    return args


class Agent:
    def __init__(self, opt, device):
        self.epsilon = opt.initial_epsilon
        self.initial_epsilon, self.final_epsilon = opt.initial_epsilon, opt.final_epsilon
        self.epsilon_decay_step = opt.num_decay_epochs

        self.batch_size = opt.batch_size
        self.update_target_rate = 10000

        self.replay_memory = deque(maxlen=opt.replay_memory_size)

        # 모델 생성
        if opt.load_model:
            model = torch.load(opt.model_path)
        else:
            model = DeepQNetwork()
        self.main_q_network = model.to(device)
        self.target_q_network = model.to(device)
        self.target_q_network.eval()
        self.update_target_q_network()

    # epsilon 값을 계산하는 함수
    def calc_epsilon(self, epoch):
        epsilon = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) * (
                opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)
        return epsilon

    # 타겟 네트워크를 업데이트하는 함수
    def update_target_q_network(self):
        self.target_q_network.load_state_dict(self.main_q_network.state_dict())

    # 미니배치를 얻는 함수
    def get_minibatch(self):
        batch = sample(self.replay_memory, min(len(self.replay_memory), opt.batch_size))
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.stack(tuple(state for state in state_batch))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.stack(tuple(state for state in next_state_batch))
        return state_batch, reward_batch, next_state_batch, done_batch


def train(opt):
    # 시드 및 장치 설정
    torch.manual_seed(opt.seed)
    device = 'cuda' if opt.use_cuda and torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.manual_seed_all(opt.seed)

    # 테트리스 환경 및 에이전트 초기화
    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
    agent = Agent(opt, device)

    optimizer = torch.optim.Adam(agent.main_q_network.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()

    state = env.reset().to(device)

    epoch = 0
    max_score, max_lines = 0, 0
    total_epoch = opt.num_epochs
    while epoch < total_epoch:
        step_count = 0  # 행동 수 초기화
        while True:
            next_steps = env.get_next_states()

            # 탐험 또는 이용 결정
            epsilon = agent.calc_epsilon(epoch)
            random_action = random() <= epsilon

            next_actions, next_states = zip(*next_steps.items())
            next_states = torch.stack(next_states).to(device)

            agent.main_q_network.eval()
            with torch.no_grad():
                predictions = agent.main_q_network(next_states)[:, 0]
            agent.main_q_network.train()
            if random_action:
                index = randint(0, len(next_steps) - 1)
            else:
                index = torch.argmax(predictions).item()

            next_state = next_states[index, :]
            action = next_actions[index]
            reward, done = env.step(action, render=opt.render)

            agent.replay_memory.append([state, reward, next_state, done])

            step_count += 1  # 행동 수 증가

            if done or step_count >= opt.max_steps:  # 종료 조건 확인
                final_score = env.score
                final_num_pieces = env.tetrominoes
                final_cleared_lines = env.cleared_lines
                state = env.reset().to(device)
                break
            else:
                state = next_state

        if epoch % agent.update_target_rate == 0:
            agent.update_target_q_network()

        epoch += 1
        state_batch, reward_batch, next_state_batch, done_batch = agent.get_minibatch()
        state_batch, reward_batch, next_state_batch = state_batch.to(device), reward_batch.to(device), next_state_batch.to(device)

        q_values = agent.main_q_network(state_batch)

        agent.main_q_network.eval()
        with torch.no_grad():
            next_prediction_batch = agent.target_q_network(next_state_batch)

        agent.main_q_network.train()
        y_batch = torch.cat(
            tuple(reward if done else reward + opt.gamma * prediction
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
            opt.num_epochs,
            final_score,
            final_cleared_lines,
            final_num_pieces,
            ))

if __name__ == "__main__":
    opt = get_args()
    train(opt)