import streamlit as st
import chess
import chess.svg
import torch
import torch.nn as nn
import torch.nn.functional as F
import base64

# --- 1. 模型架构 (保持不变) ---
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

# --- 2. 加载模型 ---
@st.cache_resource
def load_model():
    model = ChessNet()
    try:
        model.load_state_dict(torch.load("chess_master_model.pth", map_location="cpu"))
        model.eval()
    except: pass
    return model

model = load_model()

# --- 3. 辅助函数：渲染 SVG 棋盘 ---
def render_board(board):
    board_svg = chess.svg.board(board=board, size=400)
    b64 = base64.b64encode(board_svg.encode('utf-8')).decode("utf-8")
    return f'<img src="data:image/svg+xml;base64,{b64}"/>'

# --- 4. UI 逻辑 ---
st.title("🤖 Gemini Chess AI (SVG 版)")

if 'fen' not in st.session_state:
    st.session_state.fen = chess.STARTING_FEN

board = chess.Board(st.session_state.fen)

# 显示棋盘
st.write(render_board(board), unsafe_allow_html=True)

# 玩家输入走法
move_input = st.text_input("输入你的走法 (例如 e2e4):")

if st.button("走子"):
    try:
        move = chess.Move.from_uci(move_input)
        if move in board.legal_moves:
            board.push(move)
            
            # AI 思考
            if not board.is_game_over():
                with st.spinner("AI 思考中..."):
                    state = encode_board(board).unsqueeze(0)
                    with torch.no_grad():
                        logits = model(state)
                    legal_moves = list(board.legal_moves)
                    best_move = max(legal_moves, key=lambda m: logits[0, m.from_square * 64 + m.to_square])
                    board.push(best_move)
            
            st.session_state.fen = board.fen()
            st.rerun()
        else:
            st.error("非法走法！")
    except:
        st.error("输入格式错误，请使用 UCI 格式（如 e2e4）")

if board.is_game_over():
    st.success(f"游戏结束！结果: {board.result()}")
    if st.button("重开"):
        st.session_state.fen = chess.STARTING_FEN
        st.rerun()
