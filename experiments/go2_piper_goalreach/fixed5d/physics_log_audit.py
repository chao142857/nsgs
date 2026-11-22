import json, re, sys, time
from pathlib import Path
from prepare import put

ROOT=Path(__file__).resolve().parent
SERIOUS=re.compile(r"\b(?:CUDA_ERROR_(?:ILLEGAL|OUT_OF_MEMORY|LAUNCH|CONTEXT|DEVICE|UNKNOWN)\w*|device-side assert|illegal memory access|GPU has fallen off|GPU hang|GPU is lost|context is corrupted|CUDA error:|(?:contact|constraint|patch|pair|buffer|heap|stack|capacity|stream).{0,100}(?:overflow|exhausted|is too small|was insufficient|has been exceeded|have been missed|will be missed|are dropped)|(?:overflow|insufficient|not enough (?:space|memory)|increase (?:the |its )?(?:size|capacity)|dropped|missed|failed to report|discarded|lost).{0,100}(?:[Cc]ontacts?|[Cc]onstraints?|[Cc]ontact [Bb]uffers?|[Pp]atches|[Pp]airs|foundLostPairsCapacity)|(?:contacts?|constraints?|collision pairs?).{0,40}(?:are lost|were lost|discarded|dropped|missed))\b",re.I)
def audit(d):
    hits=[];warnings=[];files=[]
    for name in ('kit.log','training.log'):
        file=d/name
        if not file.exists():continue
        files.append(name)
        with file.open(errors='replace') as f:
            for line in f:
                if SERIOUS.search(line):
                    hits.append([name,line[:1700].strip()])
                elif '[Error]' in line:
                    warnings.append([name,line[:360].strip()])
    identity=d/'physical_entry.json';capacity={}
    if identity.exists():capacity=json.loads(identity.read_text()).get('mode',{}).get('buffers',{})
    proof={'passed':len(files)==2 and not hits and not warnings,'log_files':files,
           'physics_and_cuda_error_matches':len(hits),'all_kit_errors_count':len(warnings),
           'matches':hits[:35],'other_error_examples':warnings[:10], 'configured_gpu_capacities':capacity}
    put(d/'physics-log-audit.json',proof)
    return proof
def inspect_all():
    current=[]
    for d in sorted((ROOT/'benchmarks').glob('*')):
        if not (d/'supervisor-exit.json').is_file(): continue
        result=audit(d)
        sup=json.loads((d/'supervisor-exit.json').read_text())
        current.append({'directory':str(d),'execution_passed':sup['passed'],'physical_log_clear':result['passed'],
            'matched':result['physics_and_cuda_error_matches'],'other_errors':result['all_kit_errors_count']})
    put(ROOT/'benchmark-physics-audits.json',{'runs':current})
    return current
if __name__=='__main__':
    if '--watch' in sys.argv:
        while True:
            entries=inspect_all()
            if (ROOT/'benchmark-decision.json').is_file() and any('n16384' in x['directory'] for x in entries):break
            time.sleep(25)
    else: inspect_all()

