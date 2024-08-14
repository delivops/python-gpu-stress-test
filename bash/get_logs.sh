#!/bin/bash

# Get the list of pods in the nvidia-device-plugin namespace with the label app=gpu-test
pods=$(kubectl get pods -n nvidia-device-plugin -l app=gpu-test --output=jsonpath='{.items[*].metadata.name}')

# Create an array to store the pod names and last lines
results=()

# Iterate over each pod and retrieve the last line simultaneously
for pod in $pods; do
    last_line=$(kubectl exec -n nvidia-device-plugin $pod -- tail -1 perf.log &)
    results+=("$pod,$last_line")
done

# Wait for all background processes to finish
wait

# Print the table header
echo "Pod Name,Last Line"

# Print the results in a table format
for result in "${results[@]}"; do
    IFS=',' read -r pod last_line <<<"$result"
    echo "$pod,$last_line"
done
