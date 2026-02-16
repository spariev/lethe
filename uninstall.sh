#!/usr/bin/env zsh
#
# Lethe Uninstaller
# Removes Lethe and all associated files
#

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

INSTALL_DIR="${LETHE_INSTALL_DIR:-$HOME/.lethe}"
CONFIG_DIR="${LETHE_CONFIG_DIR:-$HOME/.config/lethe}"

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

check_command() {
    command -v "$1" >/dev/null 2>&1
}

prompt_read() {
    local prompt="$1"
    local var_name="$2"
    local value
    printf "%s" "$prompt" > /dev/tty
    IFS= read -r value < /dev/tty
    eval "$var_name=\$value"
}

detect_os() {
    case "$(uname -s)" in
        Linux*)
            if grep -qi microsoft /proc/version 2>/dev/null; then
                echo "wsl"
            else
                echo "linux"
            fi
            ;;
        Darwin*)
            echo "mac"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

PODMAN_MACHINE_STARTED_BY_UNINSTALL=false
PODMAN_MACHINE_NAME=""

ensure_podman_ready_on_mac() {
    if [[ "$OS" != "mac" ]]; then
        return 0
    fi
    if ! check_command podman; then
        return 1
    fi
    if podman info >/dev/null 2>&1; then
        return 0
    fi

    local candidate=""
    local state=""
    local name=""

    for name in $(podman machine list --quiet 2>/dev/null); do
        state="$(podman machine inspect "$name" --format '{{.State}}' 2>/dev/null || true)"
        if [[ "$state" == "running" || "$state" == "Running" ]]; then
            candidate="$name"
            break
        fi
        if [[ -z "$candidate" ]]; then
            candidate="$name"
        fi
    done

    if [[ -z "$candidate" ]]; then
        warn "Podman is installed, but no Podman machine exists."
        return 1
    fi

    info "Starting Podman machine ($candidate) for cleanup..."
    if podman machine start "$candidate" >/dev/null 2>&1; then
        PODMAN_MACHINE_STARTED_BY_UNINSTALL=true
        PODMAN_MACHINE_NAME="$candidate"
        return 0
    fi

    warn "Could not start Podman machine ($candidate). Skipping Podman container cleanup."
    return 1
}

cleanup_podman_machine_after_uninstall() {
    if [[ "$PODMAN_MACHINE_STARTED_BY_UNINSTALL" == "true" && -n "$PODMAN_MACHINE_NAME" ]]; then
        info "Stopping Podman machine ($PODMAN_MACHINE_NAME)..."
        podman machine stop "$PODMAN_MACHINE_NAME" >/dev/null 2>&1 || true
    fi
}

echo -e "${RED}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║                  LETHE UNINSTALLER                        ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"
WORKSPACE_DIR="${LETHE_WORKSPACE_DIR:-$HOME/lethe}"

echo ""
echo "This will remove:"
echo "  - Container (if using safe mode)"
echo "  - System service (if using native mode)"
echo "  - Installation directory: $INSTALL_DIR"
echo ""
echo "This will NOT remove:"
echo "  - Your config: $CONFIG_DIR"
echo "  - Your workspace: $WORKSPACE_DIR"
echo ""
prompt_read "Are you sure you want to uninstall Lethe? [y/N] " REPLY
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

OS=$(detect_os)

# Stop and remove systemd user service (Linux/WSL)
if [ -f "$HOME/.config/systemd/user/lethe.service" ]; then
    info "Stopping systemd user service..."
    systemctl --user stop lethe 2>/dev/null || true
    systemctl --user disable lethe 2>/dev/null || true
    rm -f "$HOME/.config/systemd/user/lethe.service"
    systemctl --user daemon-reload 2>/dev/null || true
    success "Systemd user service removed"
fi

# Stop and remove systemd system service (root installs)
if [ -f "/etc/systemd/system/lethe.service" ]; then
    info "Stopping systemd system service..."
    sudo systemctl stop lethe 2>/dev/null || systemctl stop lethe 2>/dev/null || true
    sudo systemctl disable lethe 2>/dev/null || systemctl disable lethe 2>/dev/null || true
    sudo rm -f "/etc/systemd/system/lethe.service" || rm -f "/etc/systemd/system/lethe.service"
    sudo systemctl daemon-reload 2>/dev/null || systemctl daemon-reload 2>/dev/null || true
    success "Systemd system service removed"
fi

# Stop and remove launchd service (Mac)
if [ -f "$HOME/Library/LaunchAgents/com.lethe.agent.plist" ]; then
    info "Stopping launchd service..."
    launchctl unload "$HOME/Library/LaunchAgents/com.lethe.agent.plist" 2>/dev/null || true
    rm -f "$HOME/Library/LaunchAgents/com.lethe.agent.plist"
    success "Launchd service removed"
fi

# Stop and remove container
if check_command podman; then
    podman_ready=true
    if [[ "$OS" == "mac" ]]; then
        ensure_podman_ready_on_mac || podman_ready=false
    fi

    if [[ "$podman_ready" == "true" ]]; then
        if podman ps -a --format '{{.Names}}' 2>/dev/null | grep -q '^lethe$'; then
            info "Removing Podman container..."
            podman stop lethe 2>/dev/null || true
            podman rm lethe 2>/dev/null || true
            podman rmi lethe:latest 2>/dev/null || true
            success "Podman container removed"
        fi
    fi
fi

if check_command docker; then
    if docker info >/dev/null 2>&1; then
        if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q '^lethe$'; then
            info "Removing Docker container..."
            docker stop lethe 2>/dev/null || true
            docker rm lethe 2>/dev/null || true
            docker rmi lethe:latest 2>/dev/null || true
            success "Docker container removed"
        fi
    else
        warn "Docker daemon is not reachable; skipped Docker container cleanup."
        if [[ "$OS" == "mac" ]]; then
            warn "Start Docker Desktop and rerun uninstall to remove Docker artifacts."
        fi
    fi
fi

if [[ "$OS" == "mac" ]] && check_command podman; then
    default_machine="$(podman machine list --quiet 2>/dev/null | head -n 1 || true)"
    if [[ -n "${default_machine:-}" ]]; then
        echo ""
        prompt_read "Remove Podman machine '$default_machine' too? [y/N] " remove_podman_vm
        remove_podman_vm=${remove_podman_vm:-N}
        if [[ "$remove_podman_vm" =~ ^[Yy]$ ]]; then
            info "Removing Podman machine ($default_machine)..."
            podman machine stop "$default_machine" >/dev/null 2>&1 || true
            podman machine rm -f "$default_machine" >/dev/null 2>&1 || true
            success "Podman machine removed"
        fi
    fi
fi

# Remove installation directory
if [ -d "$INSTALL_DIR" ]; then
    info "Removing installation directory..."
    rm -rf "$INSTALL_DIR"
    success "Installation directory removed"
fi

cleanup_podman_machine_after_uninstall

echo ""
success "Lethe has been uninstalled."
echo ""
echo "Your config is preserved at:"
echo "  $CONFIG_DIR - API tokens and settings"
echo ""
echo "To reinstall:"
echo "  curl -fsSL https://lethe.gg/install | zsh"
