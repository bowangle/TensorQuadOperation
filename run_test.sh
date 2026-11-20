#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

tensor_tests=(
    test_Mat_QR
    test_Mat_SVDDecomp
    test_Tensor3D
    test_tt_base
    test_mps
    test_ttn
    test_compress_svd
    test_mpo
    test_convolution
    test_analytics_construction
)
integration_tests=(
    test_indefinite_integral
    test_quantics_integrator
    test_quantics_integrator_kD
)

pids=()
names=()
run_group() {
    local group="$1"
    shift
    mkdir -p "$SCRIPT_DIR/test/$group"
    local name log_name
    for name in "$@"; do
        log_name="$name"
        # Preserve the historical filename for the TT base test log.
        if [[ "$name" == test_tt_base ]]; then
            log_name=test_TT_base
        fi
        # Redirection replaces the previous log without touching shared fixtures.
        "./build/tq_test_${name}" > "$SCRIPT_DIR/test/$group/$log_name.txt" 2>&1 &
        pids+=("$!")
        names+=("$group/$name")
    done
}

run_group TensorQuadOperation "${tensor_tests[@]}"
run_group QuanticsIntegration "${integration_tests[@]}"

# Wait for every child explicitly so a failing test cannot be hidden by bare wait.
status=0
for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
        echo "FAILED: ${names[$index]} (see its log)" >&2
        status=1
    fi
done
if [[ "$status" == 0 ]]; then
    echo "All tests completed successfully. Logs are in test/TensorQuadOperation/ and test/QuanticsIntegration/."
else
    echo "Some tests failed. Check the logs in the corresponding test subfolders." >&2
fi
exit "$status"
