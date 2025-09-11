def count_fancy_less_than_n(n: int) -> int:
    """返回小于 n 的、四进制只含 0/1 的正整数个数。"""
    if n <= 1:
        return 0

    # 转四进制（高位在前）
    digs = []
    x = n
    while x:
        digs.append(x % 4)
        x //= 4
    digs.reverse()

    ans = 0
    k = len(digs)
    for i, d in enumerate(digs):
        rem = k - i - 1
        if d == 0:
            continue
        elif d == 1:
            # 本位取 0，后面任意 0/1
            ans += 1 << rem
            # 继续走本位=1的紧致路径
        else:  # d in {2,3}
            # 本位可取 {0,1}，后面任意
            ans += 1 << (rem + 1)
            break

    # 去掉数字 0（全零）
    return max(0, ans - 1)


if __name__ == "__main__":
    n = int(input().strip())
    print(count_fancy_less_than_n(n))

