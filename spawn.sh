#!/bin/bash
# spawn.sh — launch and manage autonomous research groups
#
# Usage:
#   bash spawn.sh launch <tag> <agent:gpu> [agent:gpu ...]
#   bash spawn.sh stop <tag>
#   bash spawn.sh status
#
# Examples:
#   bash spawn.sh launch mar5 claude:0 claude:1 claude:2 claude:3
#   bash spawn.sh launch mar5 opus:0 sonnet:1 codex:2 codex:3
#   bash spawn.sh stop mar5
#   bash spawn.sh status

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
WORKTREE_DIR="${REPO_ROOT}/worktrees"
INITIAL_PROMPT="Read program.md and follow the instructions."

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

usage() {
    echo "Usage:"
    echo "  $0 launch <tag> <agent:gpu> [agent:gpu ...]"
    echo "  $0 stop <tag>"
    echo "  $0 status"
    echo ""
    echo "Examples:"
    echo "  $0 launch mar5 claude:0 claude:1 claude:2 claude:3"
    echo "  $0 launch mar5 opus:0 sonnet:1 codex:2 codex:3"
    echo "  $0 stop mar5"
    exit 1
}

setup_worker() {
    local tag="$1" gpu="$2" agent="$3"
    local branch="autoresearch/${tag}-gpu${gpu}"
    local worktree="${WORKTREE_DIR}/gpu${gpu}"
    local claude_model="${CLAUDE_MODEL:-sonnet}"

    # Create branch (must be fresh)
    cd "$REPO_ROOT"
    if git show-ref --verify --quiet "refs/heads/${branch}"; then
        echo "ERROR: Branch '${branch}' already exists. Use a different tag or clean up first." >&2
        exit 1
    fi
    git checkout -q master
    git checkout -q -b "$branch"
    git checkout -q master

    # Set up worktree
    if [ -d "$worktree" ]; then
        git worktree remove "$worktree" --force 2>/dev/null || rm -rf "$worktree"
    fi
    mkdir -p "$(dirname "$worktree")"
    git worktree add "$worktree" "$branch"

    # Symlink .venv so uv/python work
    ln -sf "${REPO_ROOT}/.venv" "${worktree}/.venv"

    # Build agent command
    case "$agent" in
        claude|sonnet|opus|haiku)
            case "$agent" in
                sonnet|opus|haiku) claude_model="$agent" ;;
            esac
            echo "cd ${worktree} && CUDA_VISIBLE_DEVICES=${gpu} claude --dangerously-skip-permissions --model ${claude_model} \"${INITIAL_PROMPT}\""
            ;;
        codex)
            echo "cd ${worktree} && CUDA_VISIBLE_DEVICES=${gpu} codex --dangerously-bypass-approvals-and-sandbox --model gpt-5.3-codex-spark \"${INITIAL_PROMPT}\""
            ;;
        *)
            echo "ERROR: Unknown agent '${agent}'. Supported: claude, sonnet, opus, haiku, codex" >&2
            git worktree remove "$worktree" --force 2>/dev/null || true
            git branch -D "$branch" 2>/dev/null || true
            exit 1
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

cmd_launch() {
    if [ $# -lt 2 ]; then
        usage
    fi

    local tag="$1"
    shift
    local specs=("$@")
    local tmux_session="autoresearch-${tag}"
    local num_workers=${#specs[@]}

    # Kill existing session
    tmux kill-session -t "$tmux_session" 2>/dev/null || true

    echo ""
    echo "============================================"
    echo "  RESEARCH GROUP: ${tag}"
    echo "============================================"
    echo "  Workers: ${num_workers}"

    local pane_cmds=()
    local labels=()
    local agents=()

    for spec in "${specs[@]}"; do
        local agent="${spec%%:*}"
        local gpu="${spec#*:}"

        echo "  - GPU ${gpu}: ${agent}"

        local cmd
        cmd=$(setup_worker "$tag" "$gpu" "$agent")
        pane_cmds+=("$cmd")
        labels+=("$spec")
        case "$agent" in
            sonnet|opus|haiku|claude) agents+=("claude") ;;
            *) agents+=("$agent") ;;
        esac
    done

    echo "  tmux: ${tmux_session}"
    echo "============================================"
    echo ""

    # Create tmux session with tiled grid
    tmux new-session -d -s "$tmux_session" -n workers \
        "${pane_cmds[0]}; echo ''; echo 'Session ended. Press any key to exit.'; read"

    for ((i=1; i<num_workers; i++)); do
        tmux split-window -t "$tmux_session:workers" \
            "${pane_cmds[$i]}; echo ''; echo 'Session ended. Press any key to exit.'; read"
        tmux select-layout -t "$tmux_session:workers" tiled
    done
    tmux select-layout -t "$tmux_session:workers" tiled

    echo "  Attach:    tmux attach -t ${tmux_session}"
    echo "  Detach:    Ctrl-b d"
    echo "  Zoom:      Ctrl-b z (toggle pane fullscreen)"
    echo "  Navigate:  Ctrl-b arrow keys"
    echo ""

    # Background watcher — nudge idle agents
    local nudge_interval=120
    echo "Watcher running (nudge interval: ${nudge_interval}s). Ctrl-C to stop watcher only."
    echo ""

    declare -A prev_hash

    while tmux has-session -t "$tmux_session" 2>/dev/null; do
        sleep "$nudge_interval"

        for ((i=0; i<num_workers; i++)); do
            tmux has-session -t "$tmux_session" 2>/dev/null || break

            local pane="${tmux_session}:workers.${i}"
            local content curr_hash

            content=$(tmux capture-pane -t "$pane" -p 2>/dev/null) || continue
            curr_hash=$(echo "$content" | md5sum | cut -d' ' -f1)

            if [ "${prev_hash[$i]:-}" != "$curr_hash" ]; then
                prev_hash[$i]="$curr_hash"
                continue
            fi

            local is_idle=false
            if echo "$content" | grep -qP '^❯\s*$'; then
                is_idle=true
            elif echo "$content" | grep -qP '^›\s*$'; then
                is_idle=true
            fi

            if [ "$is_idle" = true ]; then
                local nudge_msg="Keep going. Do not stop — continue your research loop."
                if [ "${agents[$i]}" = "codex" ]; then
                    tmux send-keys -t "$pane" "$nudge_msg" Enter
                else
                    tmux send-keys -t "$pane" "$nudge_msg" C-m
                fi
                echo "[$(date +%H:%M:%S)] Nudged pane ${i} (${labels[$i]})"
                prev_hash[$i]=""
            fi
        done
    done

    echo "tmux session ended. Watcher exiting."
}

cmd_stop() {
    if [ $# -lt 1 ]; then
        usage
    fi

    local tag="$1"
    local tmux_session="autoresearch-${tag}"

    # Kill tmux session
    if tmux kill-session -t "$tmux_session" 2>/dev/null; then
        echo "Killed tmux session '${tmux_session}'"
    else
        echo "No tmux session '${tmux_session}' found"
    fi

    # Remove worktrees
    local removed=0
    for wt in "${WORKTREE_DIR}"/gpu*; do
        [ -d "$wt" ] || continue
        git -C "$REPO_ROOT" worktree remove "$wt" --force 2>/dev/null && removed=$((removed + 1))
    done
    echo "Removed ${removed} worktree(s)"
    echo "Branches kept for review (git branch -l 'autoresearch/${tag}-*')"
}

cmd_status() {
    echo "Active tmux sessions:"
    tmux list-sessions 2>/dev/null | grep "^autoresearch-" || echo "  (none)"
    echo ""
    echo "Active worktrees:"
    git -C "$REPO_ROOT" worktree list 2>/dev/null | grep "worktrees/" || echo "  (none)"
    echo ""
    echo "Research branches:"
    git -C "$REPO_ROOT" branch -l 'autoresearch/*' 2>/dev/null || echo "  (none)"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if [ $# -lt 1 ]; then
    usage
fi

COMMAND="$1"
shift

case "$COMMAND" in
    launch) cmd_launch "$@" ;;
    stop)   cmd_stop "$@" ;;
    status) cmd_status "$@" ;;
    *)      usage ;;
esac
