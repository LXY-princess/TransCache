from typing import List, Dict

def solution(left: List[List[int]], right: List[List[int]]) -> List[int]:
    # 解析一棵树：返回 root, keys, vals, children
    def build(tree: List[List[int]]):
        n = len(tree)
        keys = [0] * n
        vals = [0] * n
        children = [[] for _ in range(n)]
        is_child = [False] * n
        for i, row in enumerate(tree):
            keys[i] = row[0]
            vals[i] = row[1]
            if len(row) > 2:
                cs = row[2:]
                children[i] = cs
                for c in cs:
                    is_child[c] = True
        root = next(i for i in range(n) if not is_child[i])
        return root, keys, vals, children

    if not left or not right:
        return []

    lroot, lkey, lval, lch = build(left)
    rroot, rkey, rval, rch = build(right)

    # 根 key 不同 => 直接空
    if lkey[lroot] != rkey[rroot]:
        return []

    res: List[int] = []

    # 拷贝整棵左/右子树（用于只在一侧出现的分支）
    def copyL(u: int):
        res.append(lkey[u]); res.append(lval[u])
        for v in lch[u]:
            copyL(v)

    def copyR(u: int):
        res.append(rkey[u]); res.append(rval[u])
        for v in rch[u]:
            copyR(v)

    # 按题意合并两节点（保证 key 相同）
    def merge(uL: int, uR: int):
        # 合并节点取 Right 的值
        res.append(rkey[uR]); res.append(rval[uR])

        # Right 子的 key -> index 映射
        rmap: Dict[int, int] = {rkey[c]: c for c in rch[uR]}

        # 先按 Left 的顺序处理
        lkeys_seen = set()
        for lc in lch[uL]:
            k = lkey[lc]
            lkeys_seen.add(k)
            if k in rmap:         # 两边都有 -> 递归合并
                merge(lc, rmap[k])
            else:                 # 只在 Left
                copyL(lc)

        # 再补上 Right 独有的孩子（保持 Right 的顺序）
        for rc in rch[uR]:
            k = rkey[rc]
            if k not in lkeys_seen:
                copyR(rc)

    merge(lroot, rroot)
    return res

if __name__ == "__main__":
    # ==== 用例1：根 key 相同，需要按规则合并 ====
    # Left:
    # 1@10
    # ├─ 2@20
    # └─ 3@30
    #     └─ 4@40
    left = [
        [1, 10, 1, 2],  # 行0: 根 1@10，孩子 -> 行1(key=2), 行2(key=3)
        [2, 20],        # 行1: 叶子 2@20
        [3, 30, 3],     # 行2: 3@30，孩子 -> 行3(key=4)
        [4, 40],        # 行3: 叶子 4@40
    ]

    # Right:
    # 1@99
    # ├─ 3@300
    # │   └─ 4@444
    # ├─ 5@50
    # └─ 2@200
    right = [
        [1, 99, 1, 2, 3],  # 行0: 根 1@99，孩子 -> 行1(key=3), 行2(key=5), 行3(key=2)
        [3, 300, 4],       # 行1: 3@300，孩子 -> 行4(key=4)
        [5, 50],           # 行2: 叶子 5@50（仅右侧有）
        [2, 200],          # 行3: 叶子 2@200（左右均有，用右值覆盖）
        [4, 444],          # 行4: 叶子 4@444（左右均有，用右值覆盖）
    ]

    # 期望：根相同 -> 合并；节点值取右边；子顺序=Left顺序 + Right独有(保持Right顺序)
    # 合并结果的前序：[1,99, 2,200, 3,300, 4,444, 5,50]
    out1 = solution(left, right)
    print(out1)  # 期望: [1, 99, 2, 200, 3, 300, 4, 444, 5, 50]

    # ==== 用例2：根 key 不同 => 返回空列表 ====
    # left2 = [
    #     [7, 70],
    # ]
    # right2 = [
    #     [8, 80],
    # ]
    # out2 = solution(left2, right2)
    # print(out2)  # 期望: []


    left2 = [
        [4,2],
        [1,15,3,2,0],
        [1,0,4],
        [6,3],
        [2,3]
    ]
    right2 = [
        [1,16,1,3],
        [0,8,2],
        [9,1],
        [1,0,4,5,6],
        [3,6],
        [2,5],
        [6,7]
    ]
    out2 = solution(left2, right2)
    print(out2)  # 期望: [1, 16, 6, 3, 1, 0, 2, 5, 3, 6, 6, 7, 4, 2, 0, 8, 9, 1]


