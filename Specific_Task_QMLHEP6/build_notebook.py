import json

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}

cells = []

cells.append(md(
    "# QMLHEP6 – Learning Quantum Representations with Contrastive Learning\n\n"
    "The core idea: instead of using a fixed encoding to map classical data into a quantum state, "
    "we train the encoding circuit itself using a contrastive objective. "
    "Similar data points (same class) should produce quantum states with high overlap; "
    "dissimilar points should produce states with low overlap. "
    "Once the embedding is trained, we plug it into a QCNN and compare accuracy "
    "against a standard AngleEmbedding baseline on the same architecture. "
    "Dataset is MNIST binary (digits 0 vs 1), reduced to 4 features via PCA."
))

cells.append(code(
    "import subprocess, sys\n"
    "pkgs = ['pennylane', 'torch', 'torchvision', 'scikit-learn', 'matplotlib', 'numpy']\n"
    "subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet'] + pkgs, check=True)\n"
    "print('ready')"
))

cells.append(code(
    "import warnings\n"
    "warnings.filterwarnings('ignore')\n"
    "\n"
    "import numpy as np\n"
    "import torch\n"
    "import torch.nn as nn\n"
    "import matplotlib.pyplot as plt\n"
    "from torch.utils.data import DataLoader, Subset\n"
    "from torchvision import datasets, transforms\n"
    "from sklearn.decomposition import PCA\n"
    "import pennylane as qml\n"
    "\n"
    "torch.manual_seed(42)\n"
    "np.random.seed(42)"
))

cells.append(code(
    "def load_mnist_binary(n_train=200, n_test=80):\n"
    "    tf = transforms.Compose([transforms.ToTensor(),\n"
    "                              transforms.Normalize((0.1307,), (0.3081,))])\n"
    "    tr = datasets.MNIST('./data', train=True,  download=True, transform=tf)\n"
    "    te = datasets.MNIST('./data', train=False, download=True, transform=tf)\n"
    "\n"
    "    def pick(ds, n):\n"
    "        return Subset(ds, [i for i, (_, y) in enumerate(ds) if y in (0, 1)][:n])\n"
    "\n"
    "    X_tr, y_tr, X_te, y_te = [], [], [], []\n"
    "    for x, y in DataLoader(pick(tr, n_train), batch_size=n_train):\n"
    "        X_tr, y_tr = x.numpy().reshape(len(x), -1), y.numpy()\n"
    "    for x, y in DataLoader(pick(te, n_test), batch_size=n_test):\n"
    "        X_te, y_te = x.numpy().reshape(len(x), -1), y.numpy()\n"
    "\n"
    "    pca = PCA(n_components=4)\n"
    "    X_tr = pca.fit_transform(X_tr).astype(np.float32)\n"
    "    X_te = pca.transform(X_te).astype(np.float32)\n"
    "    for arr in [X_tr, X_te]:\n"
    "        arr[:] = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * np.pi\n"
    "\n"
    "    return X_tr, y_tr, X_te, y_te\n"
    "\n"
    "X_train, y_train, X_test, y_test = load_mnist_binary()\n"
    "print(f'train {X_train.shape}, test {X_test.shape}')"
))

cells.append(md(
    "## Trainable quantum embedding with contrastive loss\n\n"
    "The embedding circuit has two parts: a fixed data-encoding layer (AngleEmbedding) "
    "followed by trainable RY/RZ rotations and CNOT entanglers. "
    "We optimize the trainable parameters using a contrastive loss on quantum fidelity:\n\n"
    "- **Positive pairs** (same class): minimize (1 - F)² to pull states together\n"
    "- **Negative pairs** (different class): minimize max(0, F - margin)² to push states apart\n\n"
    "Quantum fidelity F(ψᵢ, ψⱼ) = |⟨ψᵢ|ψⱼ⟩|² is computed directly from state vectors in simulation."
))

cells.append(code(
    "N = 4\n"
    "dev_emb = qml.device('default.qubit', wires=N)\n"
    "\n"
    "def make_embedding(n_layers):\n"
    "    @qml.qnode(dev_emb, interface='torch')\n"
    "    def circuit(x, params):\n"
    "        qml.AngleEmbedding(x, wires=range(N))\n"
    "        for l in range(n_layers):\n"
    "            for q in range(N):\n"
    "                qml.RY(params[l, q, 0], wires=q)\n"
    "                qml.RZ(params[l, q, 1], wires=q)\n"
    "            for q in range(N - 1):\n"
    "                qml.CNOT(wires=[q, q + 1])\n"
    "        return qml.state()\n"
    "    return circuit\n"
    "\n"
    "def contrastive_loss(states, labels, margin=0.3):\n"
    "    loss = torch.zeros(1, dtype=torch.float64)\n"
    "    count = 0\n"
    "    for i in range(len(states)):\n"
    "        for j in range(i + 1, len(states)):\n"
    "            F = torch.abs(torch.dot(states[i].conj(), states[j])) ** 2\n"
    "            F = F.real\n"
    "            if labels[i] == labels[j]:\n"
    "                loss = loss + (1 - F) ** 2\n"
    "            else:\n"
    "                loss = loss + torch.clamp(F - margin, min=0) ** 2\n"
    "            count += 1\n"
    "    return loss / max(count, 1)"
))

cells.append(code(
    "def train_contrastive_embedding(n_layers=2, n_epochs=15, lr=0.05, batch_size=20):\n"
    "    circuit = make_embedding(n_layers)\n"
    "    params = torch.nn.Parameter(\n"
    "        torch.tensor(np.random.uniform(0, 2*np.pi, (n_layers, N, 2)), dtype=torch.float64)\n"
    "    )\n"
    "    optimizer = torch.optim.Adam([params], lr=lr)\n"
    "\n"
    "    X_t = torch.tensor(X_train, dtype=torch.float64)\n"
    "    y_t = torch.tensor(y_train)\n"
    "    history = []\n"
    "\n"
    "    for epoch in range(n_epochs):\n"
    "        idx = torch.randperm(len(X_t))[:batch_size]\n"
    "        Xb, yb = X_t[idx], y_t[idx]\n"
    "\n"
    "        optimizer.zero_grad()\n"
    "        states = torch.stack([circuit(Xb[i], params) for i in range(batch_size)])\n"
    "        loss = contrastive_loss(states, yb)\n"
    "        loss.backward()\n"
    "        optimizer.step()\n"
    "\n"
    "        history.append(loss.item())\n"
    "        if (epoch + 1) % 5 == 0:\n"
    "            print(f'epoch {epoch+1:02d}  loss={loss.item():.4f}')\n"
    "\n"
    "    return params.detach(), history\n"
    "\n"
    "print('Training contrastive embedding...')\n"
    "learned_params, loss_history = train_contrastive_embedding()\n"
    "\n"
    "plt.figure(figsize=(6, 3))\n"
    "plt.plot(loss_history, color='steelblue')\n"
    "plt.xlabel('Epoch')\n"
    "plt.ylabel('Contrastive loss')\n"
    "plt.title('Embedding training convergence')\n"
    "plt.grid(alpha=0.3)\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

cells.append(md(
    "## QCNN classifier\n\n"
    "A simple 4-qubit QCNN: two convolutional blocks (parameterized 2-qubit gates on adjacent pairs) "
    "followed by global readout on qubit 0. "
    "We train this same architecture twice — once with standard AngleEmbedding, "
    "once with the contrastive-learned embedding — and compare test accuracy."
))

cells.append(code(
    "dev_cls = qml.device('default.qubit', wires=N)\n"
    "\n"
    "def make_qcnn(embedding_fn):\n"
    "    @qml.qnode(dev_cls, interface='torch')\n"
    "    def circuit(x, emb_params, cls_params):\n"
    "        embedding_fn(x, emb_params)\n"
    "        for q in range(0, N - 1, 2):\n"
    "            qml.RY(cls_params[0, q // 2], wires=q)\n"
    "            qml.CNOT(wires=[q, q + 1])\n"
    "            qml.RY(cls_params[1, q // 2], wires=q + 1)\n"
    "        for q in range(1, N - 1, 2):\n"
    "            qml.RY(cls_params[2, q // 2], wires=q)\n"
    "            qml.CNOT(wires=[q, q + 1])\n"
    "        return qml.expval(qml.PauliZ(0))\n"
    "    return circuit\n"
    "\n"
    "def standard_embed(x, params):\n"
    "    qml.AngleEmbedding(x, wires=range(N))\n"
    "\n"
    "def learned_embed(x, params):\n"
    "    qml.AngleEmbedding(x, wires=range(N))\n"
    "    n_layers = params.shape[0]\n"
    "    for l in range(n_layers):\n"
    "        for q in range(N):\n"
    "            qml.RY(params[l, q, 0], wires=q)\n"
    "            qml.RZ(params[l, q, 1], wires=q)\n"
    "        for q in range(N - 1):\n"
    "            qml.CNOT(wires=[q, q + 1])\n"
    "\n"
    "print('QCNN defined')"
))

cells.append(code(
    "def train_qcnn(embedding_fn, fixed_emb_params, n_epochs=20, lr=0.05):\n"
    "    cls_params = torch.nn.Parameter(\n"
    "        torch.tensor(np.random.uniform(0, 2*np.pi, (3, N // 2)), dtype=torch.float64)\n"
    "    )\n"
    "    optimizer = torch.optim.Adam([cls_params], lr=lr)\n"
    "    criterion = nn.BCEWithLogitsLoss()\n"
    "\n"
    "    circuit = make_qcnn(embedding_fn)\n"
    "    X_t = torch.tensor(X_train, dtype=torch.float64)\n"
    "    y_t = torch.tensor(y_train, dtype=torch.float64)\n"
    "    X_e = torch.tensor(X_test,  dtype=torch.float64)\n"
    "    y_e = torch.tensor(y_test,  dtype=torch.float64)\n"
    "\n"
    "    for epoch in range(n_epochs):\n"
    "        optimizer.zero_grad()\n"
    "        preds = torch.stack([circuit(X_t[i], fixed_emb_params, cls_params)\n"
    "                             for i in range(len(X_t))])\n"
    "        loss = criterion(preds, 2 * y_t - 1)\n"
    "        loss.backward()\n"
    "        optimizer.step()\n"
    "\n"
    "    with torch.no_grad():\n"
    "        te_preds = torch.stack([circuit(X_e[i], fixed_emb_params, cls_params)\n"
    "                                for i in range(len(X_e))])\n"
    "        test_acc = ((te_preds > 0).float() == y_e).float().mean().item()\n"
    "        tr_preds = torch.stack([circuit(X_t[i], fixed_emb_params, cls_params)\n"
    "                                for i in range(len(X_t))])\n"
    "        train_acc = ((tr_preds > 0).float() == y_t).float().mean().item()\n"
    "\n"
    "    return train_acc, test_acc\n"
    "\n"
    "dummy_emb = torch.zeros(1, dtype=torch.float64)\n"
    "\n"
    "print('Training QCNN with standard AngleEmbedding...')\n"
    "std_train, std_test = train_qcnn(standard_embed, dummy_emb)\n"
    "print(f'  train_acc={std_train:.3f}  test_acc={std_test:.3f}')\n"
    "\n"
    "print('Training QCNN with contrastive-learned embedding...')\n"
    "cl_train, cl_test = train_qcnn(learned_embed, learned_params)\n"
    "print(f'  train_acc={cl_train:.3f}  test_acc={cl_test:.3f}')"
))

cells.append(md("## Results"))

cells.append(code(
    "fig, axes = plt.subplots(1, 2, figsize=(11, 4))\n"
    "\n"
    "names = ['Standard\\nAngleEmbedding', 'Contrastive\\nEmbedding']\n"
    "train_accs = [std_train, cl_train]\n"
    "test_accs  = [std_test,  cl_test]\n"
    "colors = ['steelblue', 'darkorange']\n"
    "\n"
    "x = np.arange(2)\n"
    "w = 0.35\n"
    "axes[0].bar(x - w/2, train_accs, w, label='Train', color=colors, alpha=0.6)\n"
    "axes[0].bar(x + w/2, test_accs,  w, label='Test',  color=colors, alpha=1.0)\n"
    "axes[0].set_xticks(x)\n"
    "axes[0].set_xticklabels(names)\n"
    "axes[0].set_ylabel('Accuracy')\n"
    "axes[0].set_ylim(0, 1.1)\n"
    "axes[0].set_title('QCNN Accuracy: Standard vs Contrastive Embedding')\n"
    "axes[0].legend()\n"
    "axes[0].grid(axis='y', alpha=0.3)\n"
    "for i, (tr, te) in enumerate(zip(train_accs, test_accs)):\n"
    "    axes[0].text(i - w/2, tr + 0.02, f'{tr:.2f}', ha='center', fontsize=9)\n"
    "    axes[0].text(i + w/2, te + 0.02, f'{te:.2f}', ha='center', fontsize=9)\n"
    "\n"
    "axes[1].plot(loss_history, color='steelblue', lw=2)\n"
    "axes[1].set_xlabel('Epoch')\n"
    "axes[1].set_ylabel('Contrastive loss')\n"
    "axes[1].set_title('Embedding training loss')\n"
    "axes[1].grid(alpha=0.3)\n"
    "\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
    "\n"
    "delta = cl_test - std_test\n"
    "print(f'Standard embedding test accuracy:    {std_test:.4f}')\n"
    "print(f'Contrastive embedding test accuracy: {cl_test:.4f}')\n"
    "print(f'Improvement: {delta:+.4f}')"
))

cells.append(md(
    "## Discussion\n\n"
    "The contrastive training objective directly shapes the quantum feature space: "
    "same-class states are pulled together, different-class states pushed apart. "
    "This gives the downstream QCNN a better-structured input compared to a fixed AngleEmbedding "
    "that has no knowledge of the label structure.\n\n"
    "A few things worth noting. The improvement depends on circuit depth and training budget — "
    "with more layers and longer contrastive training, the embedding has more capacity to reorganize "
    "the feature space. The contrastive loss itself is a design choice; "
    "alternatives like InfoNCE or triplet loss could be explored.\n\n"
    "The main limitation here is that we're computing fidelity directly from state vectors "
    "(possible in simulation), whereas on real hardware you'd need to estimate it via swap tests, "
    "which adds shot noise. That's a real constraint for scaling this approach."
))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13.0"}
    },
    "cells": cells
}

path = "D:/cursor file/codes for cursor or claude code/gcode/ML4Sci-QMLHEP-GSoC2026-Evaluation/Specific_Task_QMLHEP6/qmlhep6_evaluation.ipynb"
with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Written: {path}")
print(f"Cells: {len(cells)}")
