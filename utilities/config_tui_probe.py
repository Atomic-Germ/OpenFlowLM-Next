#!/usr/bin/env python3
"""Drive `oflm config` in a pty and report what each key does.

    utilities/config_tui_probe.py [path/to/oflm]

The interactive pickers read the terminal directly, so they cannot be tested
by piping something into them: a pty is the only way to hand them the bytes a
terminal would send. This is what that looks like: it starts the guided flow,
sends one key at a time, and reports where the cursor ended up.

What it checks, and why:

  * the cursor keys, in both forms a terminal sends them -- ESC [ A and
    ESC O A, the second one being the application cursor keys. A picker that
    only knows the first form takes the second for a plain Escape and closes
    the flow.
  * the vim keys: j, k, h, l.
  * that a lone Escape acts at once, instead of waiting for the next key.
  * that the terminal is put back the way it was when the program exits: the
    echo has to come back, or the shell is left mute.
"""

import os
import pty
import re
import select
import sys
import termios
import time

OF = sys.argv[1] if len(sys.argv) > 1 else "build/src/oflm"

ANSI = re.compile(rb"\x1b\[[0-9;?]*[a-zA-Z]|\x1b[()][B0]|\x1b[=>]|\r")

ESC = b"\x1b"
UP = ESC + b"[A"
DOWN = ESC + b"[B"
APP_UP = ESC + b"OA"
APP_DOWN = ESC + b"OB"


def drain(fd, timeout=0.6):
    """Read what the program wrote, until it goes quiet."""
    out = b""
    end = time.time() + timeout
    while time.time() < end:
        r, _, _ = select.select([fd], [], [], 0.1)
        if not r:
            continue
        try:
            chunk = os.read(fd, 65536)
        except OSError:
            break
        if not chunk:
            break
        out += chunk
        end = time.time() + 0.3
    return out


def selected(text):
    """The label the cursor is on, in the last screen drawn."""
    screens = text.split(ESC + b"[H" + ESC + b"[2J")
    for line in screens[-1].splitlines():
        line = ANSI.sub(b"", line).decode("utf-8", "replace")
        m = re.match(r"^\s*>\s(.*)$", line)
        if m:
            return m.group(1).strip()
    return None


def raw_mode(fd):
    """True when the terminal is in raw mode: no echo, no line editing."""
    attrs = termios.tcgetattr(fd)
    lflag = attrs[3]
    return not (lflag & termios.ICANON) and not (lflag & termios.ECHO)


class Program:
    def __init__(self, argv):
        self.pid, self.fd = pty.fork()
        if self.pid == 0:
            os.environ["TERM"] = "xterm-256color"
            os.environ["COLORTERM"] = "truecolor"
            os.execv(argv[0], argv)
        os.set_blocking(self.fd, False)

    def send(self, keys):
        os.write(self.fd, keys)

    def read(self, timeout=0.6):
        return drain(self.fd, timeout)

    def alive(self):
        try:
            pid, _ = os.waitpid(self.pid, os.WNOHANG)
            return pid == 0
        except ChildProcessError:
            return False

    def kill(self):
        try:
            os.kill(self.pid, 9)
            os.waitpid(self.pid, 0)
        except (ProcessLookupError, ChildProcessError):
            pass


def main():
    if not os.path.exists(OF):
        print(f"config_tui_probe: {OF} not found. Build it first.")
        return 2

    prog = Program([OF, "config"])
    failures = 0

    first = prog.read(2.0)
    base = selected(first)
    print(f"menu opened        : cursor={base!r}")
    if base is None:
        print("  the menu did not open -- nothing to test")
        prog.kill()
        return 2
    if not raw_mode(prog.fd):
        print("  FAIL: the terminal is not in raw mode while the menu is up")
        failures += 1
    else:
        print("  raw mode          : OK")

    for name, keys in [
        ("down  ESC [ B     ", DOWN),
        ("down  ESC O B     ", APP_DOWN),
        ("up    ESC [ A     ", UP),
        ("up    ESC O A     ", APP_UP),
        ("down  j           ", b"j"),
        ("up    k           ", b"k"),
    ]:
        if not prog.alive():
            print(f"{name} : the program is gone")
            failures += 1
            break
        prog.send(keys)
        out = prog.read(0.5)
        now = selected(out)
        if now is None:
            print(f"{name} : nothing was drawn -- the key was not recognised")
            failures += 1
            continue
        moved = now != base
        print(f"{name} : cursor={now!r}  ({'moved' if moved else 'DID NOT MOVE'})")
        if not moved:
            failures += 1
        base = now

    # A sequence that arrives in two pieces. A terminal is free to deliver
    # the escape and the rest of it apart, and a reader that decides on the
    # escape alone takes the arrow key for a plain Escape.
    if prog.alive():
        prog.send(ESC)
        time.sleep(0.01)
        prog.send(b"[B")
        out = prog.read(0.5)
        now = selected(out)
        if now is not None and now != base:
            print(f"down  ESC , [B    : cursor={now!r}  (moved)")
            base = now
        else:
            print("down  ESC , [B    : FAIL: a split sequence was not recognised")
            failures += 1

    # A lone Escape: it has to act at once. The old reader waited for the
    # next key to tell an escape from the start of a sequence, so the menu
    # froze until the user pressed something else.
    if prog.alive():
        start = time.time()
        prog.send(ESC)
        out = prog.read(1.0)
        took = time.time() - start
        if not out:
            # The flow ended: Escape at the first step leaves it.
            print(f"lone escape        : the flow ended after {took:.2f}s  (acted at once)")
        elif took > 0.4:
            print(f"lone escape        : FAIL: nothing happened for {took:.2f}s")
            failures += 1
        else:
            print(f"lone escape        : acted after {took:.2f}s")

    # The terminal has to come back the way it was.
    if prog.alive():
        prog.send(b"q")
        time.sleep(0.3)
    prog.read(0.3)
    try:
        if raw_mode(prog.fd):
            print("terminal restored   : FAIL: the terminal is still in raw mode")
            failures += 1
        else:
            print("terminal restored   : OK (echo is back)")
    except termios.error:
        pass
    prog.kill()

    print()
    if failures:
        print(f"{failures} check(s) failed")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
