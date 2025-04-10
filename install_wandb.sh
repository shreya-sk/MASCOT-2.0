#!/bin/bash
# Script to install wandb with Python 3.12 compatibility fix

# Find the site-packages directory
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
echo "Found site-packages at: $SITE_PACKAGES"

# Copy the imp shim to site-packages
echo "Creating imp module shim..."
cat > "$SITE_PACKAGES/imp.py" << 'EOF'
"""
imp.py - A shim for the imp module removed in Python 3.12
"""
import importlib.util
import importlib.machinery
import sys

# Create a simple shim for the imp module with necessary functions
def load_source(name, pathname):
    spec = importlib.util.spec_from_file_location(name, pathname)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_module(name, file, pathname, description):
    spec = importlib.util.spec_from_file_location(name, pathname)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def find_module(name, path=None):
    try:
        spec = importlib.util.find_spec(name, path)
        if spec is None:
            return None, None, None
        return None, spec.origin, None
    except (ImportError, AttributeError):
        return None, None, None

# Add any other necessary imp functions here
EOF

echo "Installing wandb package..."
pip install --no-deps pathtools
pip install wandb==0.15.0

echo "Installation complete!"