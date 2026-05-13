import streamlit as st
from streamlit_chess_board import st_chess_board
import chess
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. 复制你原来的模型架构 (必须一致) ---
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        res = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += res
        return F.relu(out)

class ChessNet(nn.Module):
    def __init__(self, num_res_blocks=8):
        super().__init__()
        self.conv_in = nn.Conv2d(13, 128, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(128)
        self.res_layers = nn.Sequential(*[ResBlock(128) for _ in range(num_res_blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, 4096)
        )
    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.res_layers(x)
        return self.policy_head(x)

def encode_board(board):
    tensor = torch.zeros((13, 8, 8), dtype=torch.float32)
    for square, piece in board.piece_map().items():
        row, col = divmod(square, 8)
        channel = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
        tensor[channel, row, col] = 1.0
    if board.turn == chess.WHITE: tensor[12, :, :] = 1.0
    return tensor

# --- 2. 加载模型 (缓存以提高性能) ---
@st.cache_resource
def load_model():
    device = torch.device("cpu") # 网页端通常使用 CPU 推理
    model = ChessNet()
    try:
        model.load_state_dict(torch.load("chess_master_model.pth", map_location=device))
        model.eval()
    except:
        st.warning("未找到模型文件，AI 将随机走子")
    return model, device

model, device = load_model()

# --- 3. Streamlit UI 逻辑 ---
st.title("🤖 Gemini Chess AI")
st.sidebar.info("使用 PyTorch 训练的残差网络象棋助手")

# 初始化棋盘状态
if 'fen' not in st.session_state:
    st.session_state.fen = chess.STARTING_FEN

board = chess.Board(st.session_state.fen)

# 渲染棋盘并获取玩家走法
# 注意：streamlit-chess 会返回发生的走法
move_log = st_chess_board(fen=st.session_state.fen, key="board")

# 如果玩家移动了
if move_log and move_log != st.session_state.get('last_move'):
    board.push_san(move_log)
    st.session_state.last_move = move_log
    
    # AI 轮到黑棋走子
    if not board.is_game_over() and board.turn == chess.BLACK:
        with st.spinner("AI 正在思考..."):
            state = encode_board(board).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(state)
            
            legal_moves = list(board.legal_moves)
            best_move = max(legal_moves, key=lambda m: logits[0, m.from_square * 64 + m.to_square])
            board.push(best_move)
            
        st.session_state.fen = board.fen()
        st.rerun() # 强制刷新页面显示 AI 走子

if board.is_game_over():
    st.success(f"游戏结束! 结果: {board.result()}")
    if st.button("重新开始"):
        st.session_state.fen = chess.STARTING_FEN
        st.rerun()
