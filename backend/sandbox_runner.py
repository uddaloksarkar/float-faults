"""Executes one untrusted code snippet, run as a fresh subprocess per request.

Reads the snippet from stdin, runs it with a captured stdout (same wrapper the
page used when it ran client-side via Pyodide), and prints the captured
output to real stdout for the parent process to collect. Defense-in-depth
resource limits are set before exec() in case the parent's wall-clock
timeout doesn't fire for some reason.
"""
import io
import contextlib
import traceback
import resource
import sys

MEM_BYTES = 512 * 1024 * 1024   # 512MB address space
CPU_SECONDS = 25                # hard backstop; parent enforces the real per-cell timeout


def _apply_limits():
    # Defense-in-depth only -- the parent's wall-clock subprocess timeout is the
    # primary enforcement. Some platforms (e.g. macOS/Darwin) don't support
    # setrlimit for these resources at all, so failures here are non-fatal.
    for limit, value in ((resource.RLIMIT_AS, MEM_BYTES), (resource.RLIMIT_CPU, CPU_SECONDS)):
        try:
            resource.setrlimit(limit, (value, value))
        except (ValueError, OSError):
            pass


def main():
    _apply_limits()
    src = sys.stdin.read()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            exec(src, {"__name__": "__main__"})
        except SystemExit:
            pass
        except BaseException:
            traceback.print_exc(file=buf)   # print_exc() defaults to stderr -- force it into the capture

    sys.stdout.write(buf.getvalue())


if __name__ == "__main__":
    main()
