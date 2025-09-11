# from qiskit_ibm_runtime import QiskitRuntimeService

# first time run, set default
# service = QiskitRuntimeService.save_account(channel="ibm_quantum_platform",
#                                             token="qG_by5k-mlPNZiMRchM7wiSS4MaEEPnxrmd6gDxR-zNW",overwrite=True, set_as_default=True)
# print("You have", len(service.backends()), "backends available.")

# substream call
from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService()          # 若已 save_account
print("You have", len(service.backends()), "backends available.")
