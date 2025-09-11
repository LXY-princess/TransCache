from qiskit_ibm_provider import IBMProvider
from qiskit import QuantumCircuit, transpile
from qiskit.scheduler import ScheduleConfig
from qiskit.scheduler.schedule_circuit import schedule_circuit
from qiskit.compiler import assemble

provider = IBMProvider(
    channel="ibm_cloud",  # 或 "ibm_cloud"（将逐步被前者取代）
    token="qG_by5k-mlPNZiMRchM7wiSS4MaEEPnxrmd6gDxR-zNW",
    instance="pulsecache-dev"
)  # 用刚保存的 Cloud 账号
backend = provider.backends(open_pulse=True, simulator=False, operational=True)[0]
print("Using:", backend.name)

# Bell 电路
qc = QuantumCircuit(2,2)
qc.h(0); qc.cx(0,1); qc.measure([0,1],[0,1])

# 关键：先对该后端 transpile，再调度成脉冲
qc_t = transpile(qc, backend=backend, optimization_level=0)
dflt = backend.defaults()
sched_cfg = ScheduleConfig(dflt.instruction_schedule_map,
                           backend.configuration().meas_map,
                           backend.configuration().dt)
sched = schedule_circuit(qc_t, sched_cfg, backend=backend, method="asap")

# 用 assemble 打成 PulseQobj 并提交（provider 的 backend.run 支持）
qobj = assemble(sched, backend=backend, shots=256, meas_level=2, meas_return="single")
job  = backend.run(qobj)
print("Job:", job.job_id())
print("Counts:", job.result().get_counts())
