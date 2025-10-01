import qiskit, qiskit_ibm_runtime
print("qiskit =", qiskit.__version__, "| ibm-runtime =", qiskit_ibm_runtime.__version__)
# 关键模块是否存在：
import qiskit.primitives.containers as C
from qiskit_ibm_runtime import QiskitRuntimeService
print("containers OK:", C.__name__)