#!/bin/bash

# stage-specific error messages
declare -A ERROR_MESSAGES=( 
    ["docker_update"]="Failed to update or upgrade system packages. Please install Docker before continuing."
    ["docker_dependencies"]="Failed to install required dependencies. Please install Docker before continuing."
    ["docker_gpg"]="Failed to add Docker GPG key. Please install Docker before continuing."
    ["docker_repo"]="Failed to add Docker repository. Please install Docker before continuing."
    ["docker_install"]="Failed to install Docker. Please install Docker before continuing."
    ["compose_download"]="Failed to download Docker Compose binary. Please install Docker Compose before you can continue."
    ["compose_permission"]="Failed to make Docker Compose executable. Please install Docker Compose before you can continue."
    ["compose_verify"]="Docker Compose verification failed. Please install Docker Compose before you can continue."
)

# critical containers
CRITICAL_SERVICES=("cache" "db" "mqtt" "pogona_hunter")


# Docker & Docker Compose-related constants
DOCKER_GPG_URL="https://download.docker.com/linux/ubuntu/gpg"
DOCKER_KEYRING_PATH="/usr/share/keyrings/docker-archive-keyring.gpg"
DOCKER_REPO_LIST="/etc/apt/sources.list.d/docker.list"

DOCKER_CLI_PLUGIN_DIR="$HOME/.docker/cli-plugins/"
DOCKER_COMPOSE_BINARY_URL_BASE="https://github.com/docker/compose/releases/download"
DOCKER_COMPOSE_BINARY_PATH="$DOCKER_CLI_PLUGIN_DIR/docker-compose"

# pip-related constants
CONDA_ENV_NAME="PreyTouch"  
PYTHON_VERSION="3.8"
REQUIREMENTS_FILE="requirements/arena.txt"

# colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BLACK_BG_WHITE_TEXT='\033[40;97m'
NC='\033[0m' # No Color

# check if a app exita by command exists 
command_exists() {
    suppress_output command -v "$1"
}

# suppress output
suppress_output() {
    "$@" &>/dev/null
}

# handle step failures
check_step_success() {
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: $1${NC}"
        exit 1
    fi
}

# sudo permissions
keep_sudo_alive() {
    echo -e "${BLUE}Requesting sudo access...${NC}"
    sudo -v
    while true; do
        suppress_output sudo -n true
        sleep 60
        kill -0 "$$" || exit
    done 2>/dev/null &
}

install_docker() {
    echo -e "${BLUE}Docker was not found. Attempting to install Docker...${NC}"

    # Update and upgrade system packages
    echo -e "${BLUE}Step 1: Updating system repositories...${NC}"
    sudo apt update && sudo apt upgrade -y
    check_step_success "${ERROR_MESSAGES["docker_update"]}"

    # Install required dependencies
    echo -e "${BLUE}Step 2: Installing required dependencies for Docker...${NC}"
    sudo apt install -y curl gnupg lsb-release ca-certificates apt-transport-https software-properties-common
    check_step_success "${ERROR_MESSAGES["docker_dependencies"]}"

    # Add Docker GPG key (validate if it exists already)
    echo -e "${BLUE}Step 3: Adding Docker GPG key...${NC}"
    EXPECTED_FINGERPRINT="9DC858229FC7DD38854AE2D88D81803C0EBFCD88"
    if [ -f "$DOCKER_KEYRING_PATH" ]; then
        echo -e "${YELLOW}GPG key already exists at $DOCKER_KEYRING_PATH. Validating...${NC}"
        ACTUAL_FINGERPRINT=$(gpg --show-keys "$DOCKER_KEYRING_PATH" | grep -oP "[A-F0-9]{40}")
        if [ "$ACTUAL_FINGERPRINT" == "$EXPECTED_FINGERPRINT" ]; then
            echo -e "${GREEN}Existing GPG key is valid. Skipping addition.${NC}"
        else
            echo -e "${YELLOW}GPG key fingerprint mismatch. Replacing with the correct key...${NC}"
            curl -fsSL "$DOCKER_GPG_URL" | sudo gpg --dearmor -o "$DOCKER_KEYRING_PATH"
            check_step_success "${ERROR_MESSAGES["docker_gpg"]}"
            echo -e "${GREEN}Docker GPG key updated successfully.${NC}"
        fi
    else
        curl -fsSL "$DOCKER_GPG_URL" | sudo gpg --dearmor -o "$DOCKER_KEYRING_PATH"
        check_step_success "${ERROR_MESSAGES["docker_gpg"]}"
        echo -e "${GREEN}Docker GPG key added successfully.${NC}"
    fi

    # Add Docker to the system resources list
    echo -e "${BLUE}Step 4: Adding Docker repository...${NC}"
    echo "deb [arch=$(dpkg --print-architecture) signed-by=$DOCKER_KEYRING_PATH] \
https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
    sudo tee "$DOCKER_REPO_LIST" > /dev/null
    check_step_success "${ERROR_MESSAGES["docker_repo"]}"

    # Update repositories and install Docker
    echo -e "${BLUE}Step 5: Installing Docker...${NC}"
    sudo apt update && sudo apt install -y docker-ce
    check_step_success "${ERROR_MESSAGES["docker_install"]}"

    # Start and enable Docker service
    echo -e "${BLUE}Starting and enabling Docker service...${NC}"
    sudo systemctl start docker
    sudo systemctl enable docker
    check_step_success "Failed to start or enable Docker service."

    echo -e "${GREEN}Docker installed successfully.${NC}"
}

install_docker_compose() {
    echo -e "${BLUE}Docker Compose was not found. Attempting to install Docker Compose...${NC}"

    # Create plugins directory
    echo -e "${BLUE}Step 1: Creating plugins directory...${NC}"
    mkdir -p "$DOCKER_CLI_PLUGIN_DIR"
    check_step_success "Failed to create Docker Compose plugins directory. \
Ensure you have the necessary permissions."

    # Download Docker Compose binary
    echo -e "${BLUE}Step 2: Downloading Docker Compose binary...${NC}"
    ARCH=$(uname -m)
    if [[ "$ARCH" == "x86_64" ]]; then
        COMPOSE_BINARY="docker-compose-linux-x86_64"
    elif [[ "$ARCH" == "aarch64" ]]; then
        COMPOSE_BINARY="docker-compose-linux-arm64"
    else
        echo -e "${RED}Unsupported architecture: $ARCH. Exiting.${NC}"
        exit 1
    fi

    latest_stable_version=$(curl -s https://api.github.com/repos/docker/compose/releases/latest | \
    grep -oP '"tag_name": "\K(.*)(?=")')
    compose_binary_url="$DOCKER_COMPOSE_BINARY_URL_BASE/$latest_stable_version/$COMPOSE_BINARY"
    for i in {1..3}; do
        echo -e "${YELLOW}Attempting to download Docker Compose binary (Attempt $i)...${NC}"
        curl -SL "$compose_binary_url" -o "$DOCKER_COMPOSE_BINARY_PATH" && break
        sleep 5
    done
    check_step_success "${ERROR_MESSAGES["compose_download"]}"

    # Make binary executable
    echo -e "${BLUE}Step 3: Making Docker Compose executable...${NC}"
    chmod +x "$DOCKER_COMPOSE_BINARY_PATH"
    check_step_success "${ERROR_MESSAGES["compose_permission"]}"

    # Verify Docker Compose installation
    echo -e "${BLUE}Step 4: Verifying Docker Compose installation...${NC}"
    docker compose version
    check_step_success "${ERROR_MESSAGES["compose_verify"]}"

    echo -e "${GREEN}Docker Compose installed successfully.${NC}"
}

verify_docker_installation() {
    echo -e "${BLUE}Verifying Docker installation...${NC}"
    if ! command_exists docker; then
        install_docker

    else
        echo -e "${GREEN}Docker is already installed.${NC}"
    fi

    echo -e "${BLUE}Adding the current user ($USER) to the 'docker' group...${NC}"
    sudo usermod -aG docker $USER
    check_step_success "Failed to add user to 'docker' group."
}

verify_docker_compose_installation() {
    echo -e "${BLUE}Verifying Docker Compose installation...${NC}"
    if ! command_exists docker compose; then
        install_docker_compose
    else
        echo -e "${GREEN}Docker Compose is already installed.${NC}"
    fi
}

# check if a container is running
check_container_running() {
    container_name=$1
    echo "Checking if container $container_name is running..."
    # Check if the container is running
    running_container=$(docker ps --filter "name=${container_name}" --filter "status=running" --format "{{.Names}}")
    
    if [ -z "$running_container" ]; then
        echo -e "${RED}Error: $container_name is not running!${NC}, ;;; $running_container"
        echo "Fetching logs for $container_name:"
        sodu docker logs "$container_name" 2>&1 | tail -n 20  # Show the last 20 lines of logs
        exit 1
    else
        echo -e "${GREEN}Success: $container_name is running.${NC}"
    fi
}

# verify containers of critical dependencies
verify_dependencies() {
    echo -e "${BLUE}Verifying critical dependencies...${NC}"
    for container in "${CRITICAL_SERVICES[@]}"; do
        check_container_running "$container" || exit 1
    done
    echo -e "${GREEN}All critical dependencies are running.${NC}"
}

fix_docker_credentials() {
    local config_file="$HOME/.docker/config.json"

    # Ensure the Docker config file exists
    if [ -f "$config_file" ]; then
        echo "Checking Docker credentials configuration..."

        # Check for invalid or platform-specific credentials helper
        local invalid_helpers=("desktop.exe" "osxkeychain")
        for helper in "${invalid_helpers[@]}"; do
            if grep -q "\"credsStore\": \"$helper\"" "$config_file"; then
                echo "Detected invalid or platform-specific Docker credentials helper ($helper). Fixing..."
                sudo sed -i "/\"credsStore\": \"$helper\"/d" "$config_file"
                echo "Removed credentials helper: $helper"
            fi
        done

        # Check for malformed or outdated entries
        if grep -q "\"auths\": null" "$config_file"; then
            echo "Fixing malformed auths section in Docker config..."
            sudo sed -i 's/"auths": null/"auths": {}/' "$config_file"
        fi

        echo "Docker credentials configuration has been fixed."
    else
        echo "No Docker credentials configuration file found at $config_file. Skipping fix."
    fi
}


check_docker_running() {
    echo -e "${BLUE}Checking if Docker daemon is running...${NC}"
    if pgrep -x "dockerd" > /dev/null; then
        echo -e "${GREEN}Docker daemon is running.${NC}"
        return 0
    fi
    echo -e "${YELLOW}Docker daemon is not running. Attempting to start...${NC}"
    # Start Docker manually in non-systemd environments
    if [ -z "$(command -v systemctl)" ] || ! sudo systemctl is-active --quiet docker; then
        sudo dockerd &>/dev/null &
        sleep 5
        if ! pgrep -x "dockerd" > /dev/null; then
            echo -e "${RED}Failed to start Docker daemon. Please start it manually.${NC}"
            exit 1
        fi
    else
        # Start Docker with systemctl in systemd-based environments
        sudo systemctl start docker
        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to start Docker daemon with systemctl.${NC}"
            exit 1
        fi
    fi
}



start_docker_compose_services() {
    echo -e "${BLUE}Running 'docker compose up -d'...${NC}"
    docker compose up -d
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: 'docker compose up -d' failed or timed out.${NC}"
        docker compose down || echo -e "${YELLOW}Warning: Could not stop containers.${NC}"
        exit 1
    fi
    echo -e "${GREEN}Docker Compose services started successfully.${NC}"
    cd .. || exit
    verify_dependencies
}

# Run docker-compose and verify dependencies
install_docker_dependencies() {
    echo -e "${BLUE}Starting application dependencies using Docker Compose...${NC}"
    
    # Navigate to the docker directory
    cd docker/ || { echo -e "${RED}Error: 'docker/' directory not found.${NC}"; exit 1; }

    # Check if the user is in the docker group
    if ! groups $USER | grep -q '\bdocker\b'; then
        # Add the current user to the docker group
        echo -e "${BLUE}Adding the current user ($USER) to the 'docker' group...${NC}"
        sudo usermod -aG docker $USER
        check_step_success "Failed to add user to the 'docker' group."

        # Run the remaining commands in a subshell with the updated group membership
        echo -e "${YELLOW}Applying docker group changes in a subshell...${NC}"
        newgrp docker <<EOF
        $(declare -f start_docker_compose_services)
        $(declare -f verify_dependencies)
        $(declare -f check_container_running)
        declare -a CRITICAL_SERVICES=(${CRITICAL_SERVICES[@]})
        start_docker_compose_services
EOF
        exit 0
    fi

    # If the user is already in the docker group, proceed normally
    echo -e "${GREEN}User is already in the 'docker' group.${NC}"
    start_docker_compose_services
}


install_system_dependencies() {
    chmod +x Arena/scripts/arena_init.sh
    ./Arena/scripts/arena_init.sh
}

install_python_dependencies() {
    echo -e "${BLUE}Installing Python packages from "$REQUIREMENTS_FILE"...${NC}"
    # install without output to the teminal
    suppress_output pip install --upgrade pip
    suppress_output pip install -r "$REQUIREMENTS_FILE"
    check_step_success "Failed to install Python dependencies from "$REQUIREMENTS_FILE". Please resolve any issues and retry."

    echo -e "${GREEN}All Python dependencies installed successfully.${NC}"
}


install_pip_packages() {
    install_python_dependencies -y
}

check_nvidia_driver() {
    echo -e "${BLUE}Checking for NVIDIA driver...${NC}"
    if ! suppress_output nvidia-smi; then
        echo -e "${RED}You don't have an NVIDIA driver installed."
        echo -e "\n${BLUE}You can fix this by running the following command:${NC}"
        echo -e "${BLACK_BG_WHITE_TEXT}\n \n sudo apt install nvidia-driver-525 \n${NC}"
        echo -e "\n${YELLOW}❗Note: PreyTouch only tested with this driver, but other NVIDIA drivers should work as well.${NC}"
        echo -e "${YELLOW}Notice you'll have to reboot after the driver installation.${NC}"
        exit 1
    else
        echo -e "${GREEN}NVIDIA driver is installed and working.${NC}"
    fi

    # Check for cuDNN installation
    echo -e "${BLUE}Checking for cuDNN...${NC}"
    cudnn_check=$(find /usr/local -name 'libcudnn.so*' 2> /dev/null | head -n 1)
    if [ -z "$cudnn_check" ]; then
        echo -e "${RED}cuDNN is not installed or not found in the expected locations.${NC}"
        echo -e "\n${BLUE}You can install cuDNN by following the official installation guide:${NC}"
        echo -e "${BLACK_BG_WHITE_TEXT}\n \n  https://developer.nvidia.com/cudnn \n${NC}"
        exit 1
    else
        echo -e "${GREEN}cuDNN is installed. Found at: ${cudnn_check}${NC}"
    fi
}

run_preytouch() {
    cd Arena
    python api.py
}


setup_conda_env() {
    echo -e "${BLUE}Checking if Conda is installed...${NC}"

    if ! command_exists conda; then
        if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
            source "$HOME/miniconda3/etc/profile.d/conda.sh"
        elif [ -x "$HOME/miniconda3/bin/conda" ]; then
            export PATH="$HOME/miniconda3/bin:$PATH"
        fi
    fi

    if ! command_exists conda; then
        echo -e "${RED}Error: Conda is not installed. Please install Conda and try again.${NC}"
        exit 1
    fi

    echo -e "${BLUE}Checking if Conda environment '$CONDA_ENV_NAME' exists...${NC}"
    if conda info --envs | grep -q "^$CONDA_ENV_NAME"; then
        echo -e "${GREEN}Conda environment '$CONDA_ENV_NAME' already exists. Activating it...${NC}"
    else
        echo -e "${BLUE}Creating Conda environment '$CONDA_ENV_NAME' with Python $PYTHON_VERSION...${NC}"
        conda create -y -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION"
        check_step_success "Failed to create Conda environment '$CONDA_ENV_NAME'. Please check the Conda setup."
        echo -e "${GREEN}Conda environment '$CONDA_ENV_NAME' created successfully.${NC}"
    fi

    echo -e "${BLUE}Activating Conda environment '$CONDA_ENV_NAME'...${NC}"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV_NAME"
    check_step_success "Failed to activate Conda environment '$CONDA_ENV_NAME'. Please ensure Conda is properly configured."
    echo -e "${GREEN}Conda environment '$CONDA_ENV_NAME' activated successfully.${NC}"
}

keep_sudo_alive
install_system_dependencies
setup_conda_env 
verify_docker_installation
verify_docker_compose_installation
install_docker_dependencies
install_pip_packages
check_nvidia_driver
run_preytouch

echo -e "${GREEN}All dependencies are successfully installed and verified.${NC}"

