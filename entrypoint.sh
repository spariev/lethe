#!/bin/bash
set -e

# If HOST_UID is provided, create a user matching the host user
if [ -n "$HOST_UID" ]; then
    # Create user with matching UID/GID
    groupadd -g "${HOST_GID:-$HOST_UID}" hostuser 2>/dev/null || true
    useradd -u "$HOST_UID" -g "${HOST_GID:-$HOST_UID}" -m -s /bin/bash hostuser 2>/dev/null || true
    
    # Give sudo access
    echo "hostuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
    
    # Run as the host user
    exec gosu hostuser "$@"
else
    # No HOST_UID - run as root
    exec "$@"
fi
