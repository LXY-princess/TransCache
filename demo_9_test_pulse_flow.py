from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit import QuantumCircuit, transpile
from qiskit.scheduler import ScheduleConfig
from qiskit.scheduler.schedule_circuit import schedule_circuit

# 用 Cloud 凭证直连（不依赖本地保存）
svc = QiskitRuntimeService(
    channel="ibm_cloud",   # 或 "ibm_cloud"（将逐步被前者取代）
    token="qG_by5k-mlPNZiMRchM7wiSS4MaEEPnxrmd6gDxR-zNW",
    instance="pulsecache-dev"
)

backend = svc.backends(open_pulse=True, simulator=False, operational=True)[0]
print("Using:", backend.name)

# Bell 电路
qc = QuantumCircuit(2,2)
qc.h(0); qc.cx(0,1); qc.measure([0,1],[0,1])

qc_t = transpile(qc, backend=backend, optimization_level=0)

# 1.x 的调度 API：把电路排成脉冲
dflt = backend.defaults()
sched_cfg = ScheduleConfig(dflt.instruction_schedule_map,
                           backend.configuration().meas_map,
                           backend.configuration().dt)
sched = schedule_circuit(qc_t, sched_cfg, backend=backend, method="asap")
print("sched duration (dt):", sched.duration, "dt:", backend.configuration().dt)

# 直接提交脉冲（仍然走 backend.run），不需要 primitives
job = backend.run(sched, shots=256, meas_level=2, meas_return="single")
print("Job:", job.job_id())
print(job.result().get_counts())
