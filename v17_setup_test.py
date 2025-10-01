from qiskit_ibm_runtime import QiskitRuntimeService

# 只设置 IBM_QUANTUM_TOKEN 环境变量也行；没设就把 token 明文写进来
service = QiskitRuntimeService(channel="ibm_quantum_platform",
                               token="qG_by5k-mlPNZiMRchM7wiSS4MaEEPnxrmd6gDxR-zNW")  # 不带 instance，先让它连账号

# 列出你账号可见的物理后端（非模拟器）
backs = [b for b in service.backends() if not b.configuration().simulator]
for b in backs:
    try:
        props = b.properties()  # 控制面读取
        print(b.name, "| qubits:", b.configuration().n_qubits, "| last_update:", props.last_update_date)
    except Exception as e:
        print(b.name, "(no props)", e)

# first time run, set default
# from qiskit_ibm_runtime import QiskitRuntimeService
# service = QiskitRuntimeService.save_account(channel="ibm_quantum_platform",
#                                             token="qG_by5k-mlPNZiMRchM7wiSS4MaEEPnxrmd6gDxR-zNW",overwrite=True, set_as_default=True)
# print("You have", len(service.backends()), "backends available.")

# substream call
# from qiskit_ibm_runtime import QiskitRuntimeService
# service = QiskitRuntimeService()          # 若已 save_account
# print("You have", len(service.backends()), "backends available.")