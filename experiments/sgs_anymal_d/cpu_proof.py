"""Build the new bank twice, verify every ID and optionally audit the previous release."""
import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np

ROOT=Path(__file__).resolve().parent
def run_build(dest):
    subprocess.run([sys.executable,str(ROOT/'src/course.py'),str(dest)],check=True)

def digest(path):
    return sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank',type=Path,default=ROOT)
    p.add_argument('--old-source',type=Path,help='src/ from commit fd5f6c3')
    p.add_argument('--report',type=Path,required=True)
    args=p.parse_args()
    with tempfile.TemporaryDirectory() as td:
        check=Path(td)/'rebuilt';run_build(check)
        names=[f.name for f in check.iterdir() if f.is_file()]
        hashes={n:digest(check/n) for n in names}
        assert all(digest(args.bank/n)==h for n,h in hashes.items())
        train=np.load(check/'geometry.npy',allow_pickle=False)
        test=np.load(check/'holdout.npy',allow_pickle=False)
        # Full-row comparison also detects a collision after index permutation.
        def keys(a):return set(a[:,:6].copy().view('V24').ravel().tolist())
        a,b=keys(train),keys(test)
        assert len(a)==len(b)==len(train)==104000 and not a.intersection(b)
        old=None
        if args.old_source:
            oldout=Path(td)/'old';oldout.mkdir()
            subprocess.run([sys.executable,'-c',
                'import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);from course import build;build(Path(sys.argv[2]))',
                str(args.old_source.resolve()),str(oldout)],check=True)
            data=np.load(oldout/'geometry.npy')
            xy=np.linalg.norm(data[:,:2]-data[:,3:5],axis=1)
            old=dict(cross=int(np.sum(data[:,0]*data[:,3]<0)),same=int(np.sum(data[:,0]*data[:,3]>0)),
                     in_xy_radius=int((xy<=.4).sum()),content_sha256=sha256(data.tobytes()).hexdigest())
            assert (old['cross'],old['same'],old['in_xy_radius'])==(52000,52000,5199)
        result=dict(passed=True,exact_files=hashes,unique_train=len(a),unique_test=len(b),
                    train_test_overlap=len(a&b),old=old)
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
