#!/usr/bin/env bash
#
# Lethe Installer (lethe2 - local-first)
# Usage: curl -fsSL https://lethe.gg/install | bash
#
# Supports multiple LLM providers: OpenRouter, Anthropic, OpenAI
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Config
REPO_URL="https://github.com/atemerev/lethe.git"
REPO_BRANCH="lethe2"
INSTALL_DIR="${LETHE_INSTALL_DIR:-$HOME/.lethe}"
CONFIG_DIR="${LETHE_CONFIG_DIR:-$HOME/.config/lethe}"

# Provider defaults
declare -A PROVIDERS=(
    ["openrouter"]="OpenRouter (recommended - access to all models)"
    ["anthropic"]="Anthropic (Claude models directly)"
    ["openai"]="OpenAI (GPT models directly)"
)
declare -A PROVIDER_KEYS=(
    ["openrouter"]="OPENROUTER_API_KEY"
    ["anthropic"]="ANTHROPIC_API_KEY"
    ["openai"]="OPENAI_API_KEY"
)
declare -A PROVIDER_MODELS=(
    ["openrouter"]="openrouter/moonshotai/kimi-k2.5"
    ["anthropic"]="claude-opus-4-5-20250514"
    ["openai"]="gpt-5.2"
)
declare -A PROVIDER_URLS=(
    ["openrouter"]="https://openrouter.ai/keys"
    ["anthropic"]="https://console.anthropic.com/settings/keys"
    ["openai"]="https://platform.openai.com/api-keys"
)

print_header() {
    echo -e "${BLUE}"
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║                                                           ║"
    echo "║   █░░ █▀▀ ▀█▀ █░█ █▀▀                                     ║"
    echo "║   █▄▄ ██▄ ░█░ █▀█ ██▄                                     ║"
    echo "║                                                           ║"
    echo "║   Autonomous Executive Assistant (v2 - Local First)       ║"
    echo "║                                                           ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

detect_os() {
    case "$(uname -s)" in
        Linux*)
            if grep -qi microsoft /proc/version 2>/dev/null; then
                echo "wsl"
            else
                echo "linux"
            fi
            ;;
        Darwin*) echo "mac" ;;
        *) echo "unknown" ;;
    esac
}

check_command() { command -v "$1" >/dev/null 2>&1; }

maybe_sudo() {
    if [[ "$(id -u)" -eq 0 ]]; then
        "$@"
    else
        sudo "$@"
    fi
}

# Detect existing API keys
detect_api_keys() {
    DETECTED_PROVIDERS=()
    
    if [ -n "${OPENROUTER_API_KEY:-}" ]; then
        DETECTED_PROVIDERS+=("openrouter")
    fi
    if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
        DETECTED_PROVIDERS+=("anthropic")
    fi
    if [ -n "${OPENAI_API_KEY:-}" ]; then
        DETECTED_PROVIDERS+=("openai")
    fi
    
    # Also check .env file if it exists
    if [ -f "$CONFIG_DIR/.env" ]; then
        source "$CONFIG_DIR/.env" 2>/dev/null || true
        if [ -n "${OPENROUTER_API_KEY:-}" ] && [[ ! " ${DETECTED_PROVIDERS[*]} " =~ " openrouter " ]]; then
            DETECTED_PROVIDERS+=("openrouter")
        fi
        if [ -n "${ANTHROPIC_API_KEY:-}" ] && [[ ! " ${DETECTED_PROVIDERS[*]} " =~ " anthropic " ]]; then
            DETECTED_PROVIDERS+=("anthropic")
        fi
        if [ -n "${OPENAI_API_KEY:-}" ] && [[ ! " ${DETECTED_PROVIDERS[*]} " =~ " openai " ]]; then
            DETECTED_PROVIDERS+=("openai")
        fi
    fi
}

prompt_provider() {
    echo ""
    echo -e "${YELLOW}Select your LLM provider:${NC}"
    echo ""
    
    # Show detected keys
    if [ ${#DETECTED_PROVIDERS[@]} -gt 0 ]; then
        echo -e "${GREEN}Detected API keys for: ${DETECTED_PROVIDERS[*]}${NC}"
        echo ""
    fi
    
    local i=1
    for provider in openrouter anthropic openai; do
        local desc="${PROVIDERS[$provider]}"
        local detected=""
        if [[ " ${DETECTED_PROVIDERS[*]} " =~ " $provider " ]]; then
            detected="${GREEN}[key found]${NC}"
        fi
        echo -e "  $i) $desc $detected"
        ((i++))
    done
    echo ""
    
    local default_choice=1
    if [[ " ${DETECTED_PROVIDERS[*]} " =~ " anthropic " ]]; then
        default_choice=2
    elif [[ " ${DETECTED_PROVIDERS[*]} " =~ " openai " ]]; then
        default_choice=3
    fi
    
    read -p "Choose provider [1-3, default=$default_choice]: " choice < /dev/tty
    choice=${choice:-$default_choice}
    
    case $choice in
        1) SELECTED_PROVIDER="openrouter" ;;
        2) SELECTED_PROVIDER="anthropic" ;;
        3) SELECTED_PROVIDER="openai" ;;
        *) SELECTED_PROVIDER="openrouter" ;;
    esac
    
    success "Selected: $SELECTED_PROVIDER"
}

prompt_model() {
    local default_model="${PROVIDER_MODELS[$SELECTED_PROVIDER]}"
    echo ""
    echo -e "${BLUE}Default model for $SELECTED_PROVIDER: ${CYAN}$default_model${NC}"
    read -p "Press Enter to use default, or enter custom model: " custom_model < /dev/tty
    
    if [ -n "$custom_model" ]; then
        SELECTED_MODEL="$custom_model"
    else
        SELECTED_MODEL="$default_model"
    fi
    success "Model: $SELECTED_MODEL"
}

prompt_api_key() {
    local key_name="${PROVIDER_KEYS[$SELECTED_PROVIDER]}"
    local key_url="${PROVIDER_URLS[$SELECTED_PROVIDER]}"
    
    # Check if already have key
    local existing_key=""
    case $SELECTED_PROVIDER in
        openrouter) existing_key="${OPENROUTER_API_KEY:-}" ;;
        anthropic) existing_key="${ANTHROPIC_API_KEY:-}" ;;
        openai) existing_key="${OPENAI_API_KEY:-}" ;;
    esac
    
    if [ -n "$existing_key" ]; then
        echo ""
        echo -e "${GREEN}Found existing $key_name${NC}"
        API_KEY="$existing_key"
        return
    fi
    
    echo ""
    echo -e "${BLUE}$key_name required${NC}"
    echo "   Get your key at: $key_url"
    echo ""
    read -p "   $key_name: " API_KEY < /dev/tty
    if [ -z "$API_KEY" ]; then
        error "$key_name is required"
    fi
}

prompt_telegram() {
    echo ""
    echo -e "${YELLOW}Telegram Configuration:${NC}"
    echo ""
    
    echo -e "${BLUE}1. Telegram Bot Token${NC}"
    echo "   Create a bot: message @BotFather on Telegram → /newbot → copy the token"
    echo ""
    read -p "   Telegram Bot Token: " TELEGRAM_TOKEN < /dev/tty
    if [ -z "$TELEGRAM_TOKEN" ]; then
        error "Telegram token is required"
    fi
    echo ""
    
    echo -e "${BLUE}2. Your Telegram User ID${NC}"
    echo "   Message @userinfobot on Telegram - it replies with your ID (a number like 123456789)"
    echo "   This restricts the bot to only respond to you."
    echo ""
    read -p "   Telegram User ID: " TELEGRAM_USER_ID < /dev/tty
    if [ -z "$TELEGRAM_USER_ID" ]; then
        error "Telegram User ID is required (security: prevents strangers from using your bot)"
    fi
}

install_dependencies() {
    local OS=$(detect_os)
    info "Detected OS: $OS"
    
    # Install curl if missing
    if ! check_command curl; then
        info "Installing curl..."
        if [[ "$OS" == "mac" ]]; then
            brew install curl
        elif check_command apt-get; then
            maybe_sudo apt-get update -y && maybe_sudo apt-get install -y curl
        elif check_command dnf; then
            maybe_sudo dnf install -y curl
        fi
    fi
    success "curl found"
    
    # Install git if missing
    if ! check_command git; then
        info "Installing git..."
        if [[ "$OS" == "mac" ]]; then
            brew install git
        elif check_command apt-get; then
            maybe_sudo apt-get install -y git
        elif check_command dnf; then
            maybe_sudo dnf install -y git
        fi
    fi
    success "git found"
    
    # Install uv
    if ! check_command uv; then
        info "Installing uv (Python package manager)..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
        export PATH="$HOME/.cargo/bin:$PATH"
    fi
    success "uv found"
    
    # Check Python
    if ! uv python list 2>/dev/null | grep -q "3.1[1-9]"; then
        info "Installing Python 3.12 via uv..."
        uv python install 3.12
    fi
    success "Python 3.11+ available"
}

clone_repo() {
    if [ -d "$INSTALL_DIR" ]; then
        info "Updating existing installation..."
        cd "$INSTALL_DIR"
        git fetch origin
        git checkout "$REPO_BRANCH"
        git pull origin "$REPO_BRANCH"
    else
        info "Cloning Lethe..."
        git clone -b "$REPO_BRANCH" "$REPO_URL" "$INSTALL_DIR"
        cd "$INSTALL_DIR"
    fi
    success "Repository ready"
}

setup_config() {
    mkdir -p "$CONFIG_DIR"
    
    local key_name="${PROVIDER_KEYS[$SELECTED_PROVIDER]}"
    
    cat > "$CONFIG_DIR/.env" << EOF
# Lethe Configuration
# Generated by installer on $(date)

# Telegram
TELEGRAM_BOT_TOKEN=$TELEGRAM_TOKEN
TELEGRAM_ALLOWED_USER_IDS=$TELEGRAM_USER_ID

# LLM Provider
LLM_PROVIDER=$SELECTED_PROVIDER
LLM_MODEL=$SELECTED_MODEL
$key_name=$API_KEY

# Optional: Heartbeat interval (seconds, default 900 = 15 min)
# HEARTBEAT_INTERVAL=900
# HEARTBEAT_ENABLED=true

# Optional: Hippocampus (memory recall)
# HIPPOCAMPUS_ENABLED=true
EOF

    # Symlink to install dir
    ln -sf "$CONFIG_DIR/.env" "$INSTALL_DIR/.env"
    
    success "Configuration saved to $CONFIG_DIR/.env"
}

setup_systemd() {
    local OS=$(detect_os)
    
    if [[ "$OS" != "linux" && "$OS" != "wsl" ]]; then
        warn "Systemd not available on $OS, skipping service setup"
        return
    fi
    
    mkdir -p "$HOME/.config/systemd/user"
    
    cat > "$HOME/.config/systemd/user/lethe.service" << EOF
[Unit]
Description=Lethe Autonomous AI Agent
After=network.target

[Service]
Type=simple
WorkingDirectory=$INSTALL_DIR
ExecStart=$HOME/.local/bin/uv run lethe
Restart=always
RestartSec=10
Environment="PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"

[Install]
WantedBy=default.target
EOF

    systemctl --user daemon-reload
    systemctl --user enable lethe
    systemctl --user start lethe
    
    success "Systemd service installed and started"
    info "View logs: journalctl --user -u lethe -f"
}

install_deps_and_run() {
    cd "$INSTALL_DIR"
    info "Installing Python dependencies..."
    uv sync
    success "Dependencies installed"
}

main() {
    print_header
    
    # Parse args
    for arg in "$@"; do
        case $arg in
            --help|-h)
                echo "Usage: $0"
                echo ""
                echo "Installs Lethe autonomous AI assistant."
                echo "Supports OpenRouter, Anthropic, and OpenAI as LLM providers."
                exit 0
                ;;
        esac
    done
    
    # Detect existing keys
    detect_api_keys
    
    # Prompts
    prompt_provider
    prompt_model
    prompt_api_key
    prompt_telegram
    
    echo ""
    info "Installing Lethe..."
    echo ""
    
    # Install
    install_dependencies
    clone_repo
    install_deps_and_run
    setup_config
    setup_systemd
    
    echo ""
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  Lethe installed successfully!${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "  Provider: $SELECTED_PROVIDER"
    echo "  Model: $SELECTED_MODEL"
    echo ""
    echo "  Message your bot on Telegram to get started!"
    echo ""
    echo "  Useful commands:"
    echo "    View logs:     journalctl --user -u lethe -f"
    echo "    Restart:       systemctl --user restart lethe"
    echo "    Stop:          systemctl --user stop lethe"
    echo "    Config:        $CONFIG_DIR/.env"
    echo ""
}

main "$@"
