from typing import List

def solution(arg1: List[List[int]]) -> List[int]:
    """
    arg1: list of rows; each row is [key, value, child_idx_1, child_idx_2, ...]
    return: preorder traversal as [key_1, value_1, key_2, value_2, ...]
    """
    n = len(arg1)
    if n == 0:
        return []

    # 拆出 key、value、children（保持顺序）
    keys = [0] * n
    vals = [0] * n
    children = [[] for _ in range(n)]
    is_child = [False] * n  # 标记哪些行曾作为别人的子节点

    for i, row in enumerate(arg1):
        keys[i] = row[0]
        vals[i] = row[1]
        if len(row) > 2:
            kids = row[2:]
            children[i] = kids
            for c in kids:
                is_child[c] = True

    # 找根：未被标记为子节点的那个行号
    root = next(i for i in range(n) if not is_child[i])

    # 迭代前序遍历
    res: List[int] = []
    stack = [root]
    while stack:
        u = stack.pop()
        res.append(keys[u])
        res.append(vals[u])
        # 逆序压栈，保证遍历顺序与 children 列表一致
        for v in reversed(children[u]):
            stack.append(v)

    return res

if __name__ == "__main__":
    arg1 = [
        [1, 15, 1, 2, 3],  # 行0: 根 1@15，孩子是行1、行2、行3
        [6, 3],            # 行1: 叶子 6@3
        [1, 0, 4],         # 行2: 1@0，孩子是行4
        [4, 2],            # 行3: 叶子 4@2
        [2, 3],            # 行4: 叶子 2@3
    ]
    print(solution(arg1))  # 期望: [1, 15, 6, 3, 1, 0, 2, 3, 4, 2]