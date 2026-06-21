#!/bin/bash

# ==============================================================================
# EgoBench Competition - Run All Scenarios
# ==============================================================================
#
# This script runs all scenarios for the competition.
# Participants should configure their models in:
#   - config/user_agent_config.py (for user simulation)
#   - config/service_agent_config.py (for service agent)
#
# Usage:
#   # Run offline testing scenarios (default)
#   bash run_all_scenarios.sh
#
#   # Run final evaluation scenarios (retail6, retail10, kitchen4, restaurant5, order2)
#   bash run_all_scenarios.sh --final_eval
#
# Optional: Specify number of tasks per scenario
#   bash run_all_scenarios.sh --num_tasks 10
#   bash run_all_scenarios.sh --final_eval --num_tasks 10
# ==============================================================================

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default number of tasks (0 = all tasks)
NUM_tasks=0
# Default mode: offline testing
FINAL_EVAL=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_tasks)
            NUM_tasks="$2"
            shift 2
            ;;
        --final_eval)
            FINAL_EVAL=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Source environment variables if .env exists
if [ -f ".env" ]; then
    source ".env"
fi

# Your settings here
export USER_MODEL_NAME="Qwen3.5-397B-A17B"
export SERVICE_MODEL_NAME="Qwen3.5-397B-A17B"
export USER_API_BASE_URL=""
export SERVICE_API_BASE_URL=""
export API_KEY=""
export SERVICE_API_KEY=""
export VIDEO_MODE="url"


# Print configuration
echo "=========================================="
echo "EgoBench Competition - Running All Scenarios"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  User Model: $USER_MODEL_NAME"
echo "  Service Model: $SERVICE_MODEL_NAME"
echo "  User API URL: $USER_API_BASE_URL"
echo "  Service API URL: $SERVICE_API_BASE_URL"
echo "  Video Mode: $VIDEO_MODE"
echo "  Final Eval: $FINAL_EVAL"
echo "  Num tasks: $NUM_tasks"
echo ""



# Check if required environment variables are set
# if [ -z "$API_KEY" ] && [ -z "$SERVICE_API_KEY" ]; then
#     echo "Error: API_KEY or SERVICE_API_KEY environment variable is not set."
#     echo "Please set your API key in .env file or as environment variable."
#     exit 1
# fi

# Create results directory if it doesn't exist
mkdir -p results

# Function to run a scenario
run_scenario() {
    local scenario=$1
    local scenario_number=$2

    echo "Running: $scenario$scenario_number (easy mode)"

    python run/multi_agent.py \
        --scenario "$scenario" \
        --scenario_number "$scenario_number" \
        --service_model_name "$SERVICE_MODEL_NAME" \
        --multi_agent_user \
        --summary_user \
        --num_tasks "$NUM_tasks"

    echo "Completed: $scenario$scenario_number (easy mode)"
    echo ""
}

if [ "$FINAL_EVAL" = true ]; then
    # ==============================================================================
    # Final Evaluation Phase (2026.06.18 - 2026.06.25)
    # Only run the final evaluation scenarios: retail6, retail10, kitchen4, restaurant5, order2
    # ==============================================================================
    echo "=========================================="
    echo "Running Final Evaluation Scenarios"
    echo "=========================================="

    echo "=========================================="
    echo "Running Retail Scenarios (6, 10)"
    echo "=========================================="
    run_scenario "retail" 6
    run_scenario "retail" 10

    echo "=========================================="
    echo "Running Kitchen Scenario (4)"
    echo "=========================================="
    run_scenario "kitchen" 4

    echo "=========================================="
    echo "Running Restaurant Scenario (5)"
    echo "=========================================="
    run_scenario "restaurant" 5

    echo "=========================================="
    echo "Running Order Scenario (2)"
    echo "=========================================="
    run_scenario "order" 2

else
    # ==============================================================================
    # Offline Testing Phase (2026.05.18 - 2026.06.25)
    # Run all offline testing scenarios
    # ==============================================================================

    # Retail scenarios (1-10)
    echo "=========================================="
    echo "Running Retail Scenarios (1-10)"
    echo "=========================================="
    for i in $(seq 1 10); do
        run_scenario "retail" $i
    done

    # Kitchen scenarios (1-4)
    echo "=========================================="
    echo "Running Kitchen Scenarios (1-4)"
    echo "=========================================="
    for i in $(seq 1 4); do
        run_scenario "kitchen" $i
    done

    # Restaurant scenarios (1-5)
    echo "=========================================="
    echo "Running Restaurant Scenarios (1-5)"
    echo "=========================================="
    for i in $(seq 1 5); do
        run_scenario "restaurant" $i
    done

    # Order scenarios (1-2)
    echo "=========================================="
    echo "Running Order Scenarios (1-2)"
    echo "=========================================="
    for i in $(seq 1 2); do
        run_scenario "order" $i
    done
fi

echo "=========================================="
echo "All scenarios completed!"
echo "=========================================="
echo ""
echo "Results saved to: results/$SERVICE_MODEL_NAME/"
echo ""
echo "To evaluate results, run:"
echo "  bash analysis_scripts/run_eval.sh"