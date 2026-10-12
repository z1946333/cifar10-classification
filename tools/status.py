import json

from tools.launch import ROOT, is_running


def main():
    active_path = ROOT / 'logs/active.json'
    if not active_path.exists():
        print('No background job has been launched. Run START_TRAIN.cmd first.')
        return
    active = json.loads(active_path.read_text(encoding='utf-8'))
    print('Process running:', is_running(active['pid'], active.get('created')))
    print('PID:', active['pid'])
    print('Run:', active['run_name'])
    status = ROOT / 'experiments' / active['run_name'] / 'status.json'
    if status.exists():
        print(status.read_text(encoding='utf-8'))
    print('\nLOG:', active['log'])
    from pathlib import Path
    log = Path(active['log'])
    if log.exists():
        with log.open('rb') as handle:
            handle.seek(max(0, log.stat().st_size - 6000))
            print(handle.read().decode('utf-8', errors='replace'))


if __name__ == '__main__':
    main()
