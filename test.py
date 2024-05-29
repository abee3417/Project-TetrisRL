import torch
from tetris import Tetris

# 전역 변수 설정
MODEL_NAME = "Qdoub"  # 테스트하고 싶은 모델 이름 입력 (Qsimp/Qdeep/Qdoub)
EPOCH = 3000  # 테스트하고 싶은 모델의 epoch 값 입력
WIDTH = 10  # 모든 이미지의 공통 너비
HEIGHT = 20  # 모든 이미지의 공통 높이
BLOCK_SIZE = 30  # 블록 크기
FPS = 30  # 초당 프레임 수
SAVED_PATH = "trained_models"  # 저장된 모델 경로
RENDER = True  # 렌더링 여부
DROP_SPEED = 0.01  # 블록 낙하 속도

def test():
    # CUDA 사용 가능 여부에 따라 랜덤 시드 설정
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    
    # CUDA 사용 가능 여부에 따라 모델 로드
    if torch.cuda.is_available():
        model = torch.load(f"./model/{MODEL_NAME}/tetris{MODEL_NAME}_{EPOCH}")
    else:
        model = torch.load(f"./model/{MODEL_NAME}/tetris{MODEL_NAME}_{EPOCH}", map_location=lambda storage, loc: storage)
    
    model.eval()  # 모델을 평가 모드로 설정
    # Tetris 환경 설정
    env = Tetris(width=WIDTH, height=HEIGHT, block_size=BLOCK_SIZE, drop_speed=DROP_SPEED, render=RENDER)
    env.reset()  # 환경 초기화
    
    # CUDA 사용 가능 여부에 따라 모델을 CUDA로 이동
    if torch.cuda.is_available():
        model.cuda()
    
    # 게임 루프 시작
    while True:
        next_steps = env.get_next_states()  # 다음 상태 얻기
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        
        if torch.cuda.is_available():
            next_states = next_states.cuda()
            
        predictions = model(next_states)[:, 0]  # 모델 예측
        index = torch.argmax(predictions).item()
        action = next_actions[index]
        _, done = env.step(action, render=True)  # 행동 수행 및 렌더링

        # 게임 종료 조건
        if done:
            print(f'\nTraining Model: {MODEL_NAME}')
            print(f'Epoch: {EPOCH}')
            print(f'Cleared Lines: {env.cleared_lines} | Score(Reward_cum): {env.score} | Pieces(Episode_len): {env.tetrominoes}\n')
            break

if __name__ == "__main__":
    test()