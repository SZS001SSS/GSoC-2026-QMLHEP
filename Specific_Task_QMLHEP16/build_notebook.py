import json

cells = []

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}

cells.append(md(
    "# QMLHEP16 – Quantum Circuit Design with LLMs: Evaluation Task\n\n"
    "This notebook demonstrates a closed-loop framework where a Claude LLM agent designs and evaluates quantum circuits through tool calls. "
    "Three custom tools are implemented using the Orchestral AI pattern: "
    "(1) a **Meyer-Wallach entanglement measure** tool that quantifies circuit expressibility—a metric central to the QMLHEP16 project; "
    "(2) a **QNN training tool** that trains a 4-qubit variational circuit on MNIST binary classification, returning loss and accuracy for agent feedback; "
    "and (3) a **circuit architecture search** where the agent simultaneously optimizes learning rate and circuit depth. "
    "The agent reasons over returned metrics across multiple rounds and converges on a configuration that balances expressibility and trainability."
))

cells.append(code(
    "import subprocess, sys\n"
    "pkgs = ['orchestral-ai', 'pennylane', 'torch', 'torchvision',\n"
    "        'anthropic', 'scikit-learn', 'matplotlib', 'numpy']\n"
    "subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet'] + pkgs, check=True)\n"
    "print('all packages ready')"
))

cells.append(code(
    "import os, json, warnings\n"
    "warnings.filterwarnings('ignore')\n"
    "\n"
    "import numpy as np\n"
    "import matplotlib\n"
    "import matplotlib.pyplot as plt\n"
    "\n"
    "import torch\n"
    "import torch.nn as nn\n"
    "from torch.utils.data import DataLoader, Subset\n"
    "from torchvision import datasets, transforms\n"
    "\n"
    "import pennylane as qml\n"
    "\n"
    "from sklearn.decomposition import PCA\n"
    "import anthropic"
))

cells.append(code(
    "TOOLS = {}\n"
    "\n"
    "def define_tool(fn):\n"
    "    TOOLS[fn.__name__] = fn\n"
    "    return fn\n"
    "\n"
    "def get_tool_schemas():\n"
    "    import inspect\n"
    "    schemas = []\n"
    "    for name, fn in TOOLS.items():\n"
    "        sig = inspect.signature(fn)\n"
    "        props, required = {}, []\n"
    "        for pname, param in sig.parameters.items():\n"
    "            ann = param.annotation\n"
    "            ptype = 'integer' if ann == int else 'number' if ann == float else 'string'\n"
    "            props[pname] = {'type': ptype}\n"
    "            if param.default is inspect.Parameter.empty:\n"
    "                required.append(pname)\n"
    "        schemas.append({\n"
    "            'name': name,\n"
    "            'description': fn.__doc__,\n"
    "            'input_schema': {'type': 'object', 'properties': props, 'required': required}\n"
    "        })\n"
    "    return schemas\n"
    "\n"
    "print('Tool registry ready')"
))

MW_TOOL = (
    "@define_tool\n"
    "def meyer_wallach_entanglement(n_qubits: int, n_layers: int) -> str:\n"
    '    """Compute the Meyer-Wallach global entanglement measure Q for a random\n'
    "    variational quantum circuit with n_qubits qubits and n_layers layers of\n"
    "    Ry rotations and CNOT entanglers. Q in [0,1]: 0=separable, 1=maximally\n"
    "    entangled. Use this to assess circuit expressibility before training.\n"
    '    """\n'
    "    dev = qml.device('default.qubit', wires=n_qubits)\n"
    "\n"
    "    @qml.qnode(dev)\n"
    "    def circuit(params):\n"
    "        for l in range(n_layers):\n"
    "            for q in range(n_qubits):\n"
    "                qml.RY(params[l, q], wires=q)\n"
    "            for q in range(n_qubits - 1):\n"
    "                qml.CNOT(wires=[q, q + 1])\n"
    "        return qml.state()\n"
    "\n"
    "    Q_vals = []\n"
    "    for _ in range(50):\n"
    "        params = np.random.uniform(0, 2*np.pi, (n_layers, n_qubits))\n"
    "        state = circuit(params)\n"
    "        total = 0.0\n"
    "        for k in range(n_qubits):\n"
    "            keep = [i for i in range(n_qubits) if i != k]\n"
    "            rho_k = qml.math.partial_trace(np.outer(state, state.conj()), keep, n_qubits)\n"
    "            total += 1.0 - float(np.real(np.trace(rho_k @ rho_k)))\n"
    "        Q_vals.append((2.0 / n_qubits) * total)\n"
    "\n"
    "    Q_mean = float(np.mean(Q_vals))\n"
    "    Q_std  = float(np.std(Q_vals))\n"
    "    return json.dumps({\n"
    "        'n_qubits': n_qubits,\n"
    "        'n_layers': n_layers,\n"
    "        'Q_mean': round(Q_mean, 4),\n"
    "        'Q_std':  round(Q_std, 4),\n"
    "        'interpretation': 'high' if Q_mean > 0.5 else 'low'\n"
    "    })\n"
    "\n"
    "print(json.loads(meyer_wallach_entanglement(4, 2)))"
)
cells.append(code(MW_TOOL))

DATA_CELL = (
    "def load_mnist_binary(n_train=300, n_test=100):\n"
    "    tf = transforms.Compose([transforms.ToTensor(),\n"
    "                              transforms.Normalize((0.1307,), (0.3081,))])\n"
    "    train_full = datasets.MNIST('./data', train=True,  download=True, transform=tf)\n"
    "    test_full  = datasets.MNIST('./data', train=False, download=True, transform=tf)\n"
    "\n"
    "    def binary_idx(ds, n):\n"
    "        return Subset(ds, [i for i, (_, y) in enumerate(ds) if y in (0, 1)][:n])\n"
    "\n"
    "    X_tr, y_tr, X_te, y_te = [], [], [], []\n"
    "    for x, y in DataLoader(binary_idx(train_full, n_train), batch_size=n_train):\n"
    "        X_tr = x.numpy().reshape(len(x), -1); y_tr = y.numpy()\n"
    "    for x, y in DataLoader(binary_idx(test_full, n_test), batch_size=n_test):\n"
    "        X_te = x.numpy().reshape(len(x), -1); y_te = y.numpy()\n"
    "\n"
    "    pca = PCA(n_components=4)\n"
    "    X_tr = pca.fit_transform(X_tr)\n"
    "    X_te = pca.transform(X_te)\n"
    "    for arr in [X_tr, X_te]:\n"
    "        arr[:] = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * np.pi\n"
    "\n"
    "    return X_tr, y_tr, X_te, y_te\n"
    "\n"
    "X_train, y_train, X_test, y_test = load_mnist_binary()\n"
    "print(f'train={X_train.shape}, test={X_test.shape}')"
)
cells.append(code(DATA_CELL))

QNN_TOOL = (
    "N_QUBITS = 4\n"
    "dev_qnn  = qml.device('default.qubit', wires=N_QUBITS)\n"
    "_training_history = []\n"
    "\n"
    "def make_qnode(n_layers):\n"
    "    @qml.qnode(dev_qnn, interface='torch')\n"
    "    def circuit(inputs, weights):\n"
    "        qml.AngleEmbedding(inputs, wires=range(N_QUBITS))\n"
    "        for l in range(n_layers):\n"
    "            for q in range(N_QUBITS):\n"
    "                qml.RY(weights[l, q, 0], wires=q)\n"
    "                qml.RZ(weights[l, q, 1], wires=q)\n"
    "            for q in range(N_QUBITS - 1):\n"
    "                qml.CNOT(wires=[q, q + 1])\n"
    "        return qml.expval(qml.PauliZ(0))\n"
    "    return circuit\n"
    "\n"
    "@define_tool\n"
    "def train_qnn(epochs: int, learning_rate: float, n_layers: int) -> str:\n"
    '    """Train a variational QNN on MNIST binary classification (digits 0 vs 1).\n'
    "    The circuit uses AngleEmbedding on 4 qubits then n_layers of RY/RZ rotations\n"
    "    with CNOT entanglers. Returns train_loss, train_acc, test_acc. Use these\n"
    "    metrics to decide next learning_rate and n_layers values.\n"
    "    Good ranges: learning_rate 0.001-0.1, n_layers 1-4, epochs 3-10.\n"
    '    """\n'
    "    circuit = make_qnode(n_layers)\n"
    "    weights = torch.nn.Parameter(\n"
    "        torch.tensor(np.random.uniform(0, 2*np.pi, (n_layers, N_QUBITS, 2)),\n"
    "                     dtype=torch.float32)\n"
    "    )\n"
    "    optimizer = torch.optim.Adam([weights], lr=learning_rate)\n"
    "    criterion = nn.BCEWithLogitsLoss()\n"
    "    X_tr = torch.tensor(X_train, dtype=torch.float32)\n"
    "    y_tr = torch.tensor(y_train, dtype=torch.float32)\n"
    "    X_te = torch.tensor(X_test,  dtype=torch.float32)\n"
    "    y_te = torch.tensor(y_test,  dtype=torch.float32)\n"
    "\n"
    "    for _ in range(epochs):\n"
    "        optimizer.zero_grad()\n"
    "        preds = torch.stack([circuit(x, weights) for x in X_tr])\n"
    "        loss  = criterion(preds, 2*y_tr - 1)\n"
    "        loss.backward()\n"
    "        optimizer.step()\n"
    "\n"
    "    with torch.no_grad():\n"
    "        tr_p = torch.stack([circuit(x, weights) for x in X_tr])\n"
    "        te_p = torch.stack([circuit(x, weights) for x in X_te])\n"
    "        result = {\n"
    "            'epochs': epochs, 'learning_rate': learning_rate, 'n_layers': n_layers,\n"
    "            'train_loss': round(criterion(tr_p, 2*y_tr-1).item(), 4),\n"
    "            'train_acc':  round(((tr_p > 0).float() == y_tr).float().mean().item(), 4),\n"
    "            'test_acc':   round(((te_p > 0).float() == y_te).float().mean().item(), 4),\n"
    "        }\n"
    "    _training_history.append(result)\n"
    "    return json.dumps(result)\n"
    "\n"
    "print(json.loads(train_qnn(3, 0.05, 2)))"
)
cells.append(code(QNN_TOOL))

AGENT_RUNNER = (
    "def run_agent(system_prompt, user_message, max_rounds=10):\n"
    "    client   = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])\n"
    "    schemas  = get_tool_schemas()\n"
    "    messages = [{'role': 'user', 'content': user_message}]\n"
    "    log      = []\n"
    "\n"
    "    for round_i in range(max_rounds):\n"
    "        response = client.messages.create(\n"
    "            model='claude-opus-4-6',\n"
    "            max_tokens=1024,\n"
    "            system=system_prompt,\n"
    "            tools=schemas,\n"
    "            messages=messages,\n"
    "        )\n"
    "\n"
    "        if response.stop_reason == 'end_turn':\n"
    "            final = next((b.text for b in response.content if hasattr(b, 'text')), '')\n"
    "            log.append({'round': round_i, 'type': 'final', 'text': final})\n"
    "            break\n"
    "\n"
    "        tool_results = []\n"
    "        for block in response.content:\n"
    "            if block.type == 'tool_use':\n"
    "                out = TOOLS[block.name](**block.input)\n"
    "                parsed = json.loads(out) if out.startswith('{') else out\n"
    "                log.append({'round': round_i, 'tool': block.name,\n"
    "                            'input': block.input, 'output': parsed})\n"
    "                print(f'[round {round_i}] {block.name}({block.input})')\n"
    "                print(f'  -> {parsed}')\n"
    "                tool_results.append({'type': 'tool_result',\n"
    "                                     'tool_use_id': block.id, 'content': out})\n"
    "\n"
    "        messages.append({'role': 'assistant', 'content': response.content})\n"
    "        messages.append({'role': 'user',      'content': tool_results})\n"
    "\n"
    "    return log\n"
    "\n"
    "print('Agent runner ready')"
)
cells.append(code(AGENT_RUNNER))

cells.append(md("## Task 1 – Meyer-Wallach Hello World"))
cells.append(code(
    "log1 = run_agent(\n"
    "    'You are a quantum computing assistant. Call meyer_wallach_entanglement '\n"
    "    'to compare circuit expressibility. Be concise.',\n"
    "    'Compare the entanglement of a 4-qubit circuit with 1 layer versus 3 layers. '\n"
    "    'Which architecture is more expressive and why?'\n"
    ")\n"
    "print()\n"
    "print(log1[-1]['text'])"
))

cells.append(md("## Task 2 – Agent Calls QNN Training Tool"))
cells.append(code(
    "log2 = run_agent(\n"
    "    'You are a quantum ML researcher. Use train_qnn to train a QNN on MNIST. '\n"
    "    'Report the final test accuracy clearly.',\n"
    "    'Train the QNN with 5 epochs, learning_rate=0.05, n_layers=2 and report results.'\n"
    ")\n"
    "print()\n"
    "print(log2[-1]['text'])"
))

cells.append(md("## Task 3 – Agent Searches Learning Rate & Circuit Depth"))
cells.append(code(
    "_training_history.clear()\n"
    "\n"
    "log3 = run_agent(\n"
    "    'You are optimizing a QNN for MNIST binary classification. '\n"
    "    'Call train_qnn varying BOTH learning_rate (0.001 to 0.1) AND n_layers (1-4). '\n"
    "    'Use 5 epochs per run. After each result, reason about whether to increase or '\n"
    "    'decrease each parameter. Run at least 5 experiments then state the best config.',\n"
    "    'Find the best combination of learning_rate and n_layers. Be strategic.',\n"
    "    max_rounds=12\n"
    ")\n"
    "print()\n"
    "print(log3[-1]['text'])"
))

cells.append(code(
    "if _training_history:\n"
    "    lrs      = [r['learning_rate'] for r in _training_history]\n"
    "    layers   = [r['n_layers']      for r in _training_history]\n"
    "    test_acc = [r['test_acc']      for r in _training_history]\n"
    "\n"
    "    fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
    "\n"
    "    sc = axes[0].scatter(lrs, test_acc, c=layers, cmap='viridis',\n"
    "                         s=120, edgecolors='k', zorder=3)\n"
    "    axes[0].set_xscale('log')\n"
    "    axes[0].set_xlabel('Learning rate (log)')\n"
    "    axes[0].set_ylabel('Test accuracy')\n"
    "    axes[0].set_title('Agent search: LR vs Accuracy')\n"
    "    axes[0].grid(alpha=0.3)\n"
    "    plt.colorbar(sc, ax=axes[0], label='n_layers')\n"
    "\n"
    "    for i, r in enumerate(_training_history):\n"
    "        axes[1].plot(i+1, r['test_acc'], 'o',\n"
    "                     color=plt.cm.viridis(r['n_layers']/4), ms=10)\n"
    "    best_acc = max(test_acc)\n"
    "    axes[1].axhline(best_acc, ls='--', color='red', label=f'best={best_acc:.3f}')\n"
    "    axes[1].set_xlabel('Experiment #')\n"
    "    axes[1].set_ylabel('Test accuracy')\n"
    "    axes[1].set_title('Agent improvement over rounds')\n"
    "    axes[1].grid(alpha=0.3)\n"
    "    axes[1].legend()\n"
    "\n"
    "    best = max(_training_history, key=lambda r: r['test_acc'])\n"
    "    print(f\"Best: lr={best['learning_rate']}, layers={best['n_layers']}, \"\n"
    "          f\"test_acc={best['test_acc']:.4f}\")\n"
    "\n"
    "    plt.tight_layout()\n"
    "    plt.show()"
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

path = "D:/cursor file/codes for cursor or claude code/gcode/ML4Sci-QMLHEP-GSoC2026-Evaluation/Specific_Task_QMLHEP16/qmlhep16_evaluation.ipynb"
with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Written: {path}")
print(f"Cells: {len(cells)}")
