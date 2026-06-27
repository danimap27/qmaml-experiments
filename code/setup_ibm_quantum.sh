#!/bin/bash
# setup_ibm_quantum.sh — Configure IBM Quantum credentials

# IBM Quantum credentials
export IBMQ_TOKEN="6XnLT3XgK6XfLrex0UuA8wtRqNmA7HkX2nC4qVQQegFz"
export IBMQ_CRN="crn:v1:bluemix:public:quantum-computing:us-east:a/90d49da9a27e443cbff3f582bb889557:b27d908b-87b3-4dec-8dd8-cc6acefc5057::"

# Save to qiskitrc for persistent storage
mkdir -p ~/.qiskit

cat > ~/.qiskit/qiskit-ibm.json <<EOF
{
  "default-ibm-quantum": {
    "channel": "ibm_quantum",
    "token": "${IBMQ_TOKEN}",
    "instance": "${IBMQ_CRN}",
    "verify": true
  }
}
EOF

echo "IBM Quantum credentials configured successfully"
echo "Token saved to ~/.qiskit/qiskit-ibm.json"

# Test connection
echo "Testing connection to IBM Quantum..."
python3 -c "
from qiskit_ibm_runtime import QiskitRuntimeService
try:
    service = QiskitRuntimeService(channel='ibm_cloud', token='${IBMQ_TOKEN}', instance='${IBMQ_CRN}')
    backends = service.backends()
    print(f'Connected successfully! Available backends: {len(backends)}')
    for b in backends[:5]:
        print(f'  - {b.name} ({b.num_qubits} qubits)')
except Exception as e:
    print(f'Connection failed: {e}')
"
