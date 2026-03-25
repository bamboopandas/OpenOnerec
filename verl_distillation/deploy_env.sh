#!/bin/bash
# Multi-node Environment Deployment Script
# Usage: bash deploy_env.sh [--all-nodes]

set -e

SCRIPT_DIR=$(cd $(dirname $0); pwd)
PROJECT_DIR=${SCRIPT_DIR}

# Configuration
CONDA_ENV_NAME=${CONDA_ENV_NAME:-"verl"}
PYTHON_VERSION=${PYTHON_VERSION:-"3.10"}
HOSTFILE=${HOSTFILE:-"/etc/mpi/hostfile"}

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Initialize conda
init_conda() {
    for conda_sh in /root/anaconda3/etc/profile.d/conda.sh \
                    /root/miniconda3/etc/profile.d/conda.sh \
                    $HOME/anaconda3/etc/profile.d/conda.sh \
                    $HOME/miniconda3/etc/profile.d/conda.sh \
                    /opt/conda/etc/profile.d/conda.sh; do
        [ -f "$conda_sh" ] && source "$conda_sh" && return 0
    done
    command -v conda &>/dev/null
}

# Setup proxy
setup_proxy() {
    log_info "Setting up proxy..."
    unset -v http_proxy https_proxy no_proxy
    export http_proxy=http://oversea-squid2.ko.txyun:11080
    export https_proxy=http://oversea-squid2.ko.txyun:11080
    export no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com
}

# Install on local node
install_local() {
    log_info "Installing environment..."

    # Setup proxy first
    setup_proxy

    if ! init_conda; then
        log_error "Conda not found."
        exit 1
    fi

    # Configure conda for stability
    conda config --set remote_read_timeout_secs 600
    conda config --set remote_connect_timeout_secs 60
    conda config --set remote_max_retries 10

    # Create or activate conda env
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        log_warn "Environment '${CONDA_ENV_NAME}' exists, activating..."
    else
        log_info "Creating environment '${CONDA_ENV_NAME}'..."
        conda create -n ${CONDA_ENV_NAME} python=${PYTHON_VERSION} -y
    fi

    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate ${CONDA_ENV_NAME}

    log_info "Installing torch..."
    pip install torch==2.8.0
    pip install --force-reinstall torchvision==0.23.0
    pip install --force-reinstall torchaudio==2.8.0

    # Install requirements
    log_info "Installing requirements.txt..."
    pip install -r ${PROJECT_DIR}/requirements.txt

    # Install flash-attn separately
    # log_info "Installing flash-attn..."
    # pip install flash-attn==2.7.4.post1 --no-build-isolation
    pip install flash-attn --no-build-isolation

    
    # Install verl package
    log_info "Installing verl package..."
    cd ${PROJECT_DIR}
    pip install -e .

    log_info "Done!"
}

# Deploy to all nodes
deploy_all_nodes() {
    [ ! -f "${HOSTFILE}" ] && log_error "Hostfile not found: ${HOSTFILE}" && exit 1

    ALL_NODES=$(awk '!a[$1]++ {print $1}' ${HOSTFILE})
    log_info "Deploying to: ${ALL_NODES}"

    mkdir -p ./logs/deploy
    for node in ${ALL_NODES}; do
        log_info "Deploying to ${node}..."
        ssh -n ${node} "CONDA_ENV_NAME=${CONDA_ENV_NAME} bash ${SCRIPT_DIR}/deploy_env.sh" \
            > "./logs/deploy/deploy_${node}.log" 2>&1 &
    done

    wait
    log_info "Deployment completed! Check logs in ./logs/deploy/"
}

# Main
case "${1}" in
    --all-nodes) deploy_all_nodes ;;
    *) install_local ;;
esac
