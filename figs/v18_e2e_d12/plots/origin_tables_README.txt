Files:
- origin_long.csv: tidy table (Circuit, Qubits, Depth, N, Method, E2E_Latency_s, Speedup_vs_Full)
- origin_wide_latency.csv: wide latency by Method (index: Circuit, Qubits)
- origin_wide_speedup.csv: wide speedup by Method (index: Circuit, Qubits)

Method order used: ['FS+Pre+ttl+SE+ema', 'FS', 'PR', 'Full']
Speedup_vs_Full = Full_latency / method_latency; Full baseline equals 1.0.
