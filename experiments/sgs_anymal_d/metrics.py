"""Summarize independent checkpoint evaluations. Counts remain paired with their denominators."""
from argparse import ArgumentParser
from pathlib import Path
import json
from collections import Counter

CN = dict(gap='缺口',pit='坑',radial_beam='放射梁',stairs='楼梯',parallel_box='平行箱',balance_beam='平衡梁',
          stepping_stones='踏石',jump_box='跳箱',maze='迷宫',contour='起伏地形',islands='浮岛',climb_box='登箱',inverted_slope='倒坡')
def read(p):return json.loads(p.read_bytes())
def frac(s,n):return f'{s}/{n}' if n else '未测'
def table(cols,rows):
    return ['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows]
def main():
    p=ArgumentParser(description=__doc__);p.add_argument('evaluations',nargs='+',type=Path);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();a=[];b=[];sums=[]
    for directory in args.evaluations:
        meta=read(directory/'summary.json');u=meta['checkpoint'];seed=meta['seed']
        std=read(directory/'heldout_by_terrain.json'); routes=read(directory/'strict_by_terrain.json')
        assert len(std)==len(routes)==13 and sum(r['n'] for r in std)==meta['n_heldout']
        for row in std:
            def calc(d):return frac(d['success'],d['n'])
            fine=row['by_class']
            assert sum(x['n'] for x in fine.values())==row['n']
            a.append([u,CN[row['terrain']],calc(row),*[calc(fine[k]) for k in ('birth_near','same_platform','obstacle')],row['drop']+row['base_contact'],row['timeout']])
        for r in routes:
            assert r['fall']+r['timeout']+r['stable_success']+r['hold_interrupted']==r['n']
            b.append([u,CN[r['terrain']],r['n'],r['first_transfer'],r['event'],r['arrival'],r['stable_success'],r['fall'],r['timeout'],r['hold_interrupted'],f"{r['max_progress_m']:.2f}" if r['max_progress_m'] is not None else '未测'])
        sums.append(dict(update=u,seed=seed,heldout=sum(r['success'] for r in std),n=meta['n_heldout'],
              strict=sum(r['stable_success'] for r in routes),m=meta['n_strict'],
              event=sum(r['event'] for r in routes),arrival=sum(r['arrival'] for r in routes),
              fall=sum(r['fall'] for r in routes),timeout=sum(r['timeout'] for r in routes),
              same_panel=meta['panel_ids_sha256']))
    assert len({s['same_panel'] for s in sums})==1
    output=['# 单卡先导逐地形结果','',
       '独立进程读取各检查点，使用相同的留出任务 ID 和固定展示面板。重复评估服务于同一条学习曲线。','',
       '## 留出集的训练式到达','',
       '单元格为成功数/任务数。出生即近仅指位置与高度符合容差，机器人仍须保持直立、低速、有支撑 0.4 秒。','']
    output+=table(['PPO 更新','地形','全部','出生即近','其余同区','跨区','掉落/机身触地','超时'],a)
    output+=['','## 预选跨障路线的三秒严格验收','','首次换面、地形事件、正确目标面抵达和持续站稳分别计数。进展量以起终水平方向上的最大投影位移计。','']
    output+=table(['PPO 更新','地形','尝试','首次换面','地形事件','有效抵达','站稳 3 s','倒地','导航超时','保持中断','最大进展 m'],b)
    args.out.parent.mkdir(exist_ok=True,parents=True)
    args.out.write_text('\n'.join(output)+'\n')
    args.out.with_suffix('.json').write_text(json.dumps(sums,indent=2)+'\n')
if __name__=='__main__':main()
