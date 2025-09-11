from typing import List
import re

# 8 个方向
DIRS = [(-1,-1), (0,-1), (1,-1),
        (-1, 0),         (1, 0),
        (-1, 1), (0, 1), (1, 1)]

def solution(n: int, moves: List[str]) -> str:
    # 棋盘：'.' 表示空
    board = [['.' for _ in range(n)] for _ in range(n)]

    def parse_move(s: str):
        """s 的形式为 PXY（可能夹杂空格/逗号/括号等），P∈{W,B}，X,Y 为 0 索引坐标。"""
        p = s[0]
        nums = re.findall(r'\d+', s)
        if len(nums) >= 2:
            x, y = int(nums[0]), int(nums[1])
        else:
            # 兼容最原始的 'PXY'（均为一位数），如 'W31'
            x, y = int(s[1]), int(s[2])
        return p, x, y

    def apply_move(p: str, x: int, y: int):
        """把 p 落在 (x,y)，并按规则翻转。"""
        opp = 'W' if p == 'B' else 'B'
        # 题面未要求校验合法性/是否占用，这里直接覆盖为当前颜色
        board[y][x] = p

        for dx, dy in DIRS:
            cx, cy = x + dx, y + dy
            cnt = 0  # 连续对手子的数量
            # 先必须是一段连续的对手子
            while 0 <= cx < n and 0 <= cy < n and board[cy][cx] == opp:
                cx += dx
                cy += dy
                cnt += 1
            # 若这段对手子后面紧跟己方子，则把中间全部翻转
            if cnt > 0 and 0 <= cx < n and 0 <= cy < n and board[cy][cx] == p:
                fx, fy = x + dx, y + dy
                for _ in range(cnt):
                    board[fy][fx] = p
                    fx += dx
                    fy += dy

    # 依次执行每一步
    for mv in moves:
        p, x, y = parse_move(mv)
        apply_move(p, x, y)

    # 计数并按要求输出 "B W"
    B = sum(cell == 'B' for row in board for cell in row)
    W = sum(cell == 'W' for row in board for cell in row)
    return f"{B} {W}"

n = 4
moves = ["B00", "B20", "W01", "B11", "B21", "W31", "B22"]
res = solution(n, moves)
print(res)
