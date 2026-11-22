"""Own exactly one assigned GPU and one new seed under the server's data directory."""
import argparse, datetime, fcntl, json, os, signal, subprocess, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
PROJECT=ROOT.parents[1]


def write(path, record):
    tmp=path.with_suffix('.pending');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(path)


def gpu(card, query):
    return subprocess.check_output(['nvidia-smi','-i',str(card),query,'--format=csv,noheader,nounits'],text=True).strip()


def available():
    return int(next(l for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')).split()[1])*1024


def run(card,resume_at=0):
    seed=json.loads((ROOT/'protocol.json').read_text())['seeds'][card]
    with (ROOT/f'gpu{card}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not gpu(card,'--query-compute-apps=pid') and int(gpu(card,'--query-gpu=memory.used'))<256
        uuid=gpu(card,'--query-gpu=uuid')
        if resume_at:
            lane=json.loads((ROOT/f'lane{card}.json').read_text())
            out=Path(lane['out']).resolve()
            assert out.parent==ROOT/'runs' and lane['uuid']==uuid and lane['seed']==seed
            assert not json.loads((out/'exit.json').read_text())['passed']
            assert 'Inplace update to inference tensor outside InferenceMode' in (out/'error.txt').read_text()
            last=json.loads((out/'updates.jsonl').read_text().splitlines()[-1])
            assert last['update']==resume_at and (out/f'checkpoint-{resume_at:04d}.pt').is_file()
            assert (out/'evaluations'/f'u{resume_at:04d}'/'summary.json').is_file()
            attempt=out/'attempts'/f'pre-resume-{int(time.time())}'
            attempt.mkdir(parents=True)
            write(attempt/'lane.json',lane)
            for name in ('error.txt','exit.json','train.log','kit.log','resources.json','portable'):
                original=out/name
                if original.exists():original.rename(attempt/name)
        else:
            out=ROOT/'runs'/f'gpu{card}-s{seed}-{int(time.time())}'
            out.mkdir(parents=True)
        cpus=sorted(os.sched_getaffinity(0))[4*card:4*card+4];assert len(cpus)==4
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,CUDA_DEVICE_ORDER='PCI_BUS_ID',PXR_WORK_THREAD_LIMIT='2',
                 OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
                 PYTHONDONTWRITEBYTECODE='1',VK_DRIVER_FILES='/etc/vulkan/icd.d/nvidia_icd.json',
                 VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json',LD_PRELOAD=str(PROJECT/'runtime/env/lib/libstdc++.so.6'))
        env.pop('DISPLAY',None); env.pop('WAYLAND_DISPLAY',None);env.pop('SGS_ANYMALD_BIRTH_OVERRIDES',None)
        args=[str(PROJECT/'runtime/with-env'),'taskset','-c',','.join(map(str,cpus)),str(PROJECT/'runtime/env/bin/python'),
              '-Bu',str(ROOT/'src/train.py'),str(card),str(seed),str(out)]
        if resume_at:args.extend(['--resume-at',str(resume_at)])
        reason=None;pos={};rss_floor=available()
        with (out/'train.log').open('w') as log:
            child=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                                    cwd=ROOT,env=env,start_new_session=True)
            write(ROOT/f'lane{card}.json',dict(card=card,uuid=uuid,seed=seed,out=str(out),pid=child.pid,cpus=cpus,phase='restoring' if resume_at else 'initializing',resume_at=resume_at))
            try:
                while child.poll() is None:
                    avail=available();rss_floor=min(avail,rss_floor)
                    vram=int(gpu(card,'--query-gpu=memory.used'))
                    others=set(gpu(card,'--query-compute-apps=pid').split())-{str(child.pid)}
                    if others: reason='unexpected compute peer on claimed GPU'
                    if avail<11*2**30:reason='host memory protection'
                    for f in (out/'kit.log',out/'train.log'):
                        if f.exists():
                            with f.open(errors='replace') as stream:
                                stream.seek(pos.get(f,0));block=stream.read().lower();pos[f]=stream.tell()
                            if any(t in block for t in ('buffer overflow detected','missed interactions','out of memory','illegal memory access','unable to find a free aggregate')):reason='physics/CUDA resource error'
                    write(out/'resources.json',dict(when=datetime.datetime.now().astimezone().isoformat(),mem_available=avail,vram_mib=vram,min_available=rss_floor))
                    if reason:break
                    time.sleep(5)
            finally:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=20)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            ok=child.returncode==0 and (out/'result.json').is_file()
            write(out/'exit.json',dict(passed=ok,code=child.returncode,stop=reason))
            return 0 if ok else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('card',type=int,choices=range(4))
    parser.add_argument('--resume-at',type=int,default=0)
    args=parser.parse_args()
    sys.exit(run(args.card,args.resume_at))
