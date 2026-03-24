import pennylane as qml
from pennylane import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

dev1 = qml.device('default.qubit', wires=5)

@qml.qnode(dev1)
def circuit_1():
    for i in range(5):
        qml.Hadamard(wires=i)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[3, 4])
    qml.SWAP(wires=[0, 4])
    qml.RX(np.pi / 2, wires=2)
    return [qml.expval(qml.PauliZ(i)) for i in range(5)]

dev2 = qml.device('default.qubit', wires=5)

@qml.qnode(dev2)
def circuit_2():
    qml.Hadamard(wires=0)
    qml.RX(np.pi / 3, wires=1)
    qml.Hadamard(wires=2)
    qml.Hadamard(wires=3)
    qml.Hadamard(wires=4)
    qml.CSWAP(wires=[4, 0, 2])
    qml.CSWAP(wires=[4, 1, 3])
    qml.Hadamard(wires=4)
    return qml.expval(qml.PauliZ(wires=4))


print(qml.draw(circuit_1)())
result1 = circuit_1()
print("Circuit 1 Z expectation values:", [float(v) for v in result1])

print(qml.draw(circuit_2)())
result2 = circuit_2()
overlap = (1 + float(result2)) / 2
print(f"Circuit 2 ancilla <Z>: {float(result2):.4f}")
print(f"State overlap: {overlap:.4f}")


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.bar([f"q{i}" for i in range(5)], [float(v) for v in result1], color='steelblue', edgecolor='black')
ax.set_ylim(-1.2, 1.2)
ax.set_ylabel("<Z>")
ax.set_title("Circuit 1: Z expectation values")
ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')

ax = axes[1]
ax.pie([overlap, 1 - overlap],
       labels=['Overlap', 'Non-overlap'],
       autopct='%1.1f%%',
       colors=['#55A868', '#C44E52'])
ax.set_title(f"Circuit 2: State overlap = {overlap:.4f}")

plt.tight_layout()
plt.savefig('Task_I_Quantum_Computing/task1_results.png', dpi=150, bbox_inches='tight')
print("Saved task1_results.png")
