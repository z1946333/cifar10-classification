"""Launch one background training job and record its log and PID."""
import ctypes
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from ctypes import wintypes
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def process_start_time(pid):
    if os.name == 'nt':
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        kernel.GetProcessTimes.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return None
        code = ctypes.c_ulong()
        try:
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value != 259:
                return None
            created, exited, system, user = (wintypes.FILETIME() for _ in range(4))
            if not kernel.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited), ctypes.byref(system), ctypes.byref(user)):
                return None
            return (created.dwHighDateTime << 32) | created.dwLowDateTime
        finally:
            kernel.CloseHandle(handle)
    try:
        return int(Path('/proc/{}/stat'.format(pid)).read_text().rsplit(')', 1)[1].split()[19])
    except (FileNotFoundError, ProcessLookupError):
        return None


def is_running(pid, created=None):
    current = process_start_time(pid)
    return current is not None and created is not None and current == created


@contextmanager
def launcher_lock(path):
    # The OS releases this lock if the owner crashes. The file may safely remain.
    with Path(path).open('a+b') as handle:
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise BlockingIOError('Another launcher is starting.') from exc
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def main():
    logs = ROOT / 'logs'
    logs.mkdir(exist_ok=True)
    active_path = logs / 'active.json'
    lock_path = logs / 'launch.lock'
    with launcher_lock(lock_path):
        if active_path.exists():
            previous = json.loads(active_path.read_text(encoding='utf-8'))
            if is_running(previous['pid'], previous.get('created')):
                print('A training job is already running/queued. No duplicate was started.')
                print('Log: ' + previous['log'])
                return
        name = 'resnet18_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        log_path = logs / (name + '.log')
        command = [sys.executable, '-u', '-m', 'scripts.train', '--device', 'cuda', '--wait-for-gpu', '--run-name', name]
        options = {'creationflags': subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == 'nt' else {'start_new_session': True}
        with log_path.open('w', encoding='utf-8') as output:
            process = subprocess.Popen(command, cwd=str(ROOT), stdin=subprocess.DEVNULL,
                                       stdout=output, stderr=subprocess.STDOUT, **options)
        active = {'pid': process.pid, 'created': process_start_time(process.pid), 'run_name': name, 'log': str(log_path),
                  'status': str(ROOT / 'experiments' / name / 'status.json')}
        temporary = active_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(active, indent=2), encoding='utf-8')
        os.replace(temporary, active_path)
        print('Training queued. PID: ' + str(process.pid))
        print('Run: ' + name)
        print('Log: ' + str(log_path))
        print('The job waits for at least 1800 MiB free GPU memory. Existing tasks are never stopped.')
        print('Keep Windows awake. Use SHOW_STATUS.cmd to check progress.')


if __name__ == '__main__':
    main()
