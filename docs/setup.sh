#!/usr/bin/env bash
set -e

echo "=== Step 1: Checking command line tools (this may open a popup)..."
if ! xcode-select -p >/dev/null 2>&1; then
  echo "Command Line Tools not found. Installing..."
  xcode-select --install || true
  echo
  echo "If you saw a popup, please click 'Install' and wait until it finishes."
  echo "Then run this script again."
  exit 0
fi

echo
echo "=== Step 2: Installing Homebrew (the package manager) if needed... ==="
if ! command -v brew >/dev/null 2>&1; then
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

  # Add Homebrew to PATH for Apple Silicon and Intel
  if [ -d "/opt/homebrew/bin" ]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
  elif [ -d "/usr/local/Homebrew/bin" ]; then
    eval "$(/usr/local/Homebrew/bin/brew shellenv)"
  fi
else
  echo "Homebrew already installed. Updating..."
  brew update
fi

# Make sure brew is on PATH in this script
if command -v brew >/dev/null 2>&1; then
  eval "$(brew shellenv)"
fi

echo
echo "=== Step 3: Installing Ruby via Homebrew... ==="
brew install ruby

# Ensure Homebrew Ruby is on PATH for this script run
if [ -d "/opt/homebrew/opt/ruby/bin" ]; then
  export PATH="/opt/homebrew/opt/ruby/bin:$PATH"
elif [ -d "/usr/local/opt/ruby/bin" ]; then
  export PATH="/usr/local/opt/ruby/bin:$PATH"
fi

echo
echo "Ruby version:"
ruby -v

echo
echo "=== Step 4: Installing Jekyll and Bundler gems (user-local)... ==="
gem install --user-install bundler jekyll

# Add gem user path to PATH so 'jekyll' is found
GEM_BIN_DIR="$(ruby -e 'print Gem.user_dir')/bin"
export PATH="$GEM_BIN_DIR:$PATH"

echo
echo "Gem bin directory is: $GEM_BIN_DIR"
echo "If you want this permanently, add this line to your shell config (~/.zshrc):"
echo "    export PATH=\"$GEM_BIN_DIR:\$PATH\""
echo

# === Project-specific: point this to your Jekyll site ===
# For your FastFlowLM repo, the Jekyll site is under docs/
PROJECT_DIR="$HOME/flm/FastFlowLM"
JEKYLL_DIR="$PROJECT_DIR/docs"

if [ ! -d "$JEKYLL_DIR" ]; then
  echo "WARNING: Expected Jekyll directory not found at:"
  echo "  $JEKYLL_DIR"
  echo "Please edit this script and set PROJECT_DIR/JEKYLL_DIR correctly."
  exit 1
fi

echo "=== Step 5: Installing site dependencies and starting Jekyll server... ==="
cd "$JEKYLL_DIR"

if [ -f "Gemfile" ]; then
  echo "Gemfile found. Running bundle install..."
  bundle install
  echo
  echo "Starting Jekyll server with Bundler..."
  bundle exec jekyll serve
else
  echo "No Gemfile found. Running jekyll serve directly..."
  jekyll serve --watch --incremental --livereload
fi

echo
echo "If the server started, open this in your browser:"
echo "  http://127.0.0.1:4000"