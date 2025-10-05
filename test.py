import random, itertools

lam = 5.0  # 每秒期望2次
inter_arrivals = [random.expovariate(lam) for _ in range(5)]  # 5个到达间隔
arrival_times = list(itertools.accumulate(inter_arrivals))    # 累加成到达时刻

print(inter_arrivals)
print(arrival_times)