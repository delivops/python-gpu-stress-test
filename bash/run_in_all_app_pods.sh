#!/bin/bash

python_script=$(dirname "$0")"../main.py"
device="cpu" # "cpu" or "cuda"
label="app=gpu-test"
namespace="nvidia-device-plugin"

# Get all pods with label app=gpu-test in the nvidia-device-plugin namespace
pods=$(kubectl get pods -n $namespace -l $label -o jsonpath='{.items[*].metadata.name}')

if [ -z "$pods" ]; then
    echo "No pods found with label $label in the $namespace namespace"
    exit 1
fi

# Loop through each pod
for pod in $pods; do
    echo "Processing pod: $pod"

    # Copy the main.py file to the pod
    kubectl cp main.py nvidia-device-plugin/$pod:/main.py

    # Execute the main.py file in the pod
    kubectl exec -n nvidia-device-plugin $pod -- python main.py

    # Create perf.log file in the pod
    kubectl exec -n nvidia-device-plugin $pod -- bash -c "touch perf.log"

    # Run main.py as a daemon
    kubectl exec -n nvidia-device-plugin $pod -- bash -c "nohup python main.py --device $device > /dev/null 2>&1 &"

    echo "Started GPU stress test on pod: $pod"
done

echo "GPU stress test started on all pods"

# ==================================================================================================

while [ 1 ]; do
    # Wait for 30 seconds
    echo "Waiting for 30 seconds to get logs..."
    sleep 30
    bash $(dirname "$0")/get_logs.sh
    echo "========================================"
done

# ==================================================================================================
