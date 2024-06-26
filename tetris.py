import numpy as np
import pygame
from pygame.locals import *
from matplotlib import style
import torch
import random
import time

style.use("ggplot")


class Tetris:
    pieces = [  # 테트로미노 (테트리스 블록) 정의
        [[1, 1],
         [1, 1]],

        [[0, 2, 0],
         [2, 2, 2]],

        [[0, 3, 3],
         [3, 3, 0]],

        [[4, 4, 0],
         [0, 4, 4]],

        [[5, 5, 5, 5]],

        [[0, 0, 6],
         [6, 6, 6]],

        [[7, 0, 0],
         [7, 7, 7]]
    ]

    piece_colors = [  # 테트로미노 (테트리스 블록) 색깔 정의
        (0, 0, 0),
        (255, 255, 0),
        (147, 88, 254),
        (54, 175, 144),
        (255, 0, 0),
        (102, 217, 238),
        (254, 151, 32),
        (0, 0, 255)
    ]

    def __init__(self, height=20, width=10, block_size=20, drop_speed=1.0, render=True, test_mode=False):
        self.height = height
        self.width = width
        self.block_size = block_size
        self.drop_speed = drop_speed  # 블록이 떨어지는 속도
        self.bg_color = (255, 255, 255)
        self.text_color = (0, 0, 0)
        self.extra_board_width = 200
        self.screen_width = self.width * self.block_size + self.extra_board_width
        self.screen_height = self.height * self.block_size
        self.render_mode = render
        self.test_mode = test_mode

        self.reset()

        if self.render_mode:
            # pygame 초기화
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Tetris RL")
            self.clock = pygame.time.Clock()

    def reset(self):
        self.board = [[0] * self.width for _ in range(self.height)]  # 테트리스 보드 정의
        self.score = 0
        self.tetrominoes = 0
        self.cleared_lines = 0
        self.bag = list(range(len(self.pieces)))
        random.shuffle(self.bag)
        self.ind = self.bag.pop()
        self.piece = [row[:] for row in self.pieces[self.ind]]
        self.current_pos = {"x": self.width // 2 - len(self.piece[0]) // 2, "y": 0}
        self.gameover = False
        self.last_drop_time = time.time()  # 마지막으로 블록이 떨어진 시간
        return self.get_state_properties(self.board)

    def rotate(self, piece):
        num_rows_orig = num_cols_new = len(piece)
        num_rows_new = len(piece[0])
        rotated_array = []

        for i in range(num_rows_new):
            new_row = [0] * num_cols_new
            for j in range(num_cols_new):
                new_row[j] = piece[(num_rows_orig - 1) - j][i]
            rotated_array.append(new_row)
        return rotated_array

    def get_state_properties(self, board):
        lines_cleared, board = self.check_cleared_rows(board)
        holes = self.get_holes(board)
        bumpiness, height = self.get_bumpiness_and_height(board)

        return torch.FloatTensor([lines_cleared, holes, bumpiness, height])

    def get_holes(self, board):
        num_holes = 0
        for col in zip(*board):
            row = 0
            while row < self.height and col[row] == 0:
                row += 1
            num_holes += len([x for x in col[row + 1:] if x == 0])
        return num_holes

    def get_bumpiness_and_height(self, board):
        board = np.array(board)
        mask = board != 0
        invert_heights = np.where(mask.any(axis=0), np.argmax(mask, axis=0), self.height)
        heights = self.height - invert_heights
        total_height = np.sum(heights)
        currs = heights[:-1]
        nexts = heights[1:]
        diffs = np.abs(currs - nexts)
        total_bumpiness = np.sum(diffs)
        return total_bumpiness, total_height

    def get_next_states(self):
        states = {}
        piece_id = self.ind
        curr_piece = [row[:] for row in self.piece]
        if piece_id == 0:  # O piece
            num_rotations = 1
        elif piece_id == 2 or piece_id == 3 or piece_id == 4:
            num_rotations = 2
        else:
            num_rotations = 4

        for i in range(num_rotations):
            valid_xs = self.width - len(curr_piece[0])
            for x in range(valid_xs + 1):
                piece = [row[:] for row in curr_piece]
                pos = {"x": x, "y": 0}
                while not self.check_collision(piece, pos):
                    pos["y"] += 1
                self.truncate(piece, pos)
                board = self.store(piece, pos)
                states[(x, i)] = self.get_state_properties(board)
            curr_piece = self.rotate(curr_piece)
        return states

    def get_current_board_state(self):
        board = [x[:] for x in self.board]
        for y in range(len(self.piece)):
            for x in range(len(self.piece[y])):
                board[y + self.current_pos["y"]][x + self.current_pos["x"]] = self.piece[y][x]
        return board

    def new_piece(self):
        if not len(self.bag):
            self.bag = list(range(len(self.pieces)))
            random.shuffle(self.bag)
        self.ind = self.bag.pop()
        self.piece = [row[:] for row in self.pieces[self.ind]]
        self.current_pos = {"x": self.width // 2 - len(self.piece[0]) // 2,
                            "y": 0
                            }
        if self.check_collision(self.piece, self.current_pos):
            self.gameover = True

    def check_collision(self, piece, pos):
        future_y = pos["y"] + 1
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if future_y + y > self.height - 1 or self.board[future_y + y][pos["x"] + x] and piece[y][x]:
                    return True
        return False

    def truncate(self, piece, pos):
        gameover = False
        last_collision_row = -1
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if self.board[pos["y"] + y][pos["x"] + x] and piece[y][x]:
                    if y > last_collision_row:
                        last_collision_row = y

        if pos["y"] - (len(piece) - last_collision_row) < 0 and last_collision_row > -1:
            while last_collision_row >= 0 and len(piece) > 1:
                gameover = True
                last_collision_row = -1
                del piece[0]
                for y in range(len(piece)):
                    for x in range(len(piece[y])):
                        if self.board[pos["y"] + y][pos["x"] + x] and piece[y][x] and y > last_collision_row:
                            last_collision_row = y
        return gameover

    def store(self, piece, pos):
        board = [x[:] for x in self.board]
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if piece[y][x] and not board[y + pos["y"]][x + pos["x"]]:
                    board[y + pos["y"]][x + pos["x"]] = piece[y][x]
        return board

    def check_cleared_rows(self, board):
        to_delete = []
        for i, row in enumerate(board[::-1]):
            if 0 not in row:
                to_delete.append(len(board) - 1 - i)
        if len(to_delete) > 0:
            board = self.remove_row(board, to_delete)
        return len(to_delete), board

    def remove_row(self, board, indices):
        for i in indices[::-1]:
            del board[i]
            board = [[0 for _ in range(self.width)]] + board
        return board

    def step(self, action, render=True):
        x, num_rotations = action
        self.current_pos = {"x": x, "y": 0}
        for _ in range(num_rotations):
            self.piece = self.rotate(self.piece)

        while not self.check_collision(self.piece, self.current_pos):
            if self.drop_speed > 0:  # 0이상일때만 테트리스 블록속도 조절
                current_time = time.time()
                if current_time - self.last_drop_time > self.drop_speed:
                    self.current_pos["y"] += 1
                    self.last_drop_time = current_time
            else:
                self.current_pos["y"] += 1
            if render and self.render_mode:
                self.render()

        overflow = self.truncate(self.piece, self.current_pos)
        if overflow:
            self.gameover = True

        self.board = self.store(self.piece, self.current_pos)

        lines_cleared, self.board = self.check_cleared_rows(self.board)
        score = 1 + (lines_cleared ** 2) * self.width
        self.score += score
        self.tetrominoes += 1
        self.cleared_lines += lines_cleared
        if not self.gameover:
            self.new_piece()
        if self.gameover:
            self.score -= 2
            if self.test_mode:
                self.render_game_over()  # 게임 오버 상태 렌더링

        return score, self.gameover

    def render(self):
        self.screen.fill(self.bg_color)

        for y in range(self.height):
            for x in range(self.width):
                color = self.piece_colors[self.board[y][x]]
                pygame.draw.rect(self.screen, color,
                                 (x * self.block_size, y * self.block_size, self.block_size, self.block_size))
                pygame.draw.rect(self.screen, (0, 0, 0),
                                 (x * self.block_size, y * self.block_size, self.block_size, self.block_size), 1)

        for y in range(len(self.piece)):
            for x in range(len(self.piece[y])):
                if self.piece[y][x]:
                    color = self.piece_colors[self.piece[y][x]]
                    pygame.draw.rect(self.screen, color,
                                     ((x + self.current_pos["x"]) * self.block_size,
                                      (y + self.current_pos["y"]) * self.block_size,
                                      self.block_size, self.block_size))
                    pygame.draw.rect(self.screen, (0, 0, 0),
                                     ((x + self.current_pos["x"]) * self.block_size,
                                      (y + self.current_pos["y"]) * self.block_size,
                                      self.block_size, self.block_size), 1)

        # 텍스트 렌더링
        font = pygame.font.SysFont("Consolas", 20)
        text_title = font.render(f"Tetris RL", True, self.text_color)
        text_score = font.render(f"Score: {self.score}", True, self.text_color)
        text_pieces = font.render(f"Pieces: {self.tetrominoes}", True, self.text_color)
        text_lines = font.render(f"Lines: {self.cleared_lines}", True, self.text_color)

        self.screen.blit(text_title, (self.width * self.block_size + 10, 10))
        self.screen.blit(text_score, (self.width * self.block_size + 10, 70))
        self.screen.blit(text_pieces, (self.width * self.block_size + 10, 100))
        self.screen.blit(text_lines, (self.width * self.block_size + 10, 130))

        pygame.display.flip()

    def render_game_over(self):
        # test.py 실행 중일때 게임 끝나는 화면 구현
        self.screen.fill(self.bg_color)

        for y in range(self.height):
            for x in range(self.width):
                color = self.piece_colors[self.board[y][x]]
                pygame.draw.rect(self.screen, color,
                                 (x * self.block_size, y * self.block_size, self.block_size, self.block_size))
                pygame.draw.rect(self.screen, (0, 0, 0),
                                 (x * self.block_size, y * self.block_size, self.block_size, self.block_size), 1)

        for y in range(len(self.piece)):
            for x in range(len(self.piece[y])):
                if self.piece[y][x]:
                    color = self.piece_colors[self.piece[y][x]]
                    pygame.draw.rect(self.screen, color,
                                     ((x + self.current_pos["x"]) * self.block_size,
                                      (y + self.current_pos["y"]) * self.block_size,
                                      self.block_size, self.block_size))
                    pygame.draw.rect(self.screen, (0, 0, 0),
                                     ((x + self.current_pos["x"]) * self.block_size,
                                      (y + self.current_pos["y"]) * self.block_size,
                                      self.block_size, self.block_size), 1)

        font = pygame.font.SysFont("Consolas", 20)
        game_over_text = font.render("Game over!", True, self.text_color)
        text_score = font.render(f"Score: {self.score}", True, self.text_color)
        text_pieces = font.render(f"Pieces: {self.tetrominoes}", True, self.text_color)
        text_lines = font.render(f"Lines: {self.cleared_lines}", True, self.text_color)

        self.screen.blit(game_over_text, (self.width * self.block_size + 10, 10))
        self.screen.blit(text_score, (self.width * self.block_size + 10, 70))
        self.screen.blit(text_pieces, (self.width * self.block_size + 10, 100))
        self.screen.blit(text_lines, (self.width * self.block_size + 10, 130))

        pygame.display.flip()
        
        while self.test_mode and self.gameover:
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    pygame.quit()
                    return
            self.clock.tick(10)