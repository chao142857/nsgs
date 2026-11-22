"""Draw the actual generated collision surfaces and coordinates; requires Pillow, NumPy."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import course as c

PAPER=(250,250,246); INK=(26,43,53); MUTED=(66,86,94); VOID=(61,68,78)
GREEN=(0,119,97); RED=(176,31,61); BLUE=(27,62,148); WHITE=(255,255,255)
TITLES=dict(zip(c.TERRAIN_NAMES,['Gap','Pit','Radial beam','Stairs','Parallel boxes','Balance beam',
    'Stepping stones','Jump boxes','Maze','Contour','Floating islands','Climbing boxes','Inverted slope']))
BANDS=[(-.8,(47,115,151)),(-.3,(94,155,164)),(0,(181,202,175)),(.4,(231,210,148)),(.85,(223,157,104)),(1.55,(163,90,86))]
def font(s):
    for name in ('/System/Library/Fonts/Supplemental/Arial.ttf','DejaVuSans.ttf'):
        try:return ImageFont.truetype(name,s)
        except OSError:pass
    return ImageFont.load_default()
def txt(d,xy,s,size,fill=INK,anchor='la'):
    d.text(xy,s,font=font(size),fill=fill,anchor=anchor)
def colors(a):
    vals,col=zip(*BANDS)
    out=np.uint8(np.stack([np.interp(a,vals,[v[i] for v in col]) for i in range(3)],axis=-1))
    out[a<c.DROP+.01]=VOID
    # Subtle top-face boundaries preserve the exact 15-cm cell geometry.
    edge=np.r_[np.zeros_like(a[:1],dtype=bool),abs(np.diff(a,axis=0))>.025]
    out[edge&(a>c.DROP+.01)]=(out[edge&(a>c.DROP+.01)]*.82).astype(np.uint8)
    return out
def arrow(d,a,b,col,width=3,dash=False):
    a=np.array(a,float);b=np.array(b,float);dist=np.linalg.norm(b-a)
    if dist<1:return
    vec=(b-a)/dist;perp=np.array([-vec[1],vec[0]])
    if dash:
        for offset in np.arange(0,max(0,dist-8),11):d.line([tuple(a+vec*offset),tuple(a+vec*min(offset+6,dist-8))],col,width=width)
    else:d.line([tuple(a),tuple(b)],col,width=width)
    d.polygon([tuple(b),tuple(b-vec*11+perp*4),tuple(b-vec*11-perp*4)],fill=col)
def marker(d,xy,role,r=4):
    x,y=xy
    if role=='S':
        d.polygon([(x,y-r-1),(x-r-1,y+r),(x+r+1,y+r)],fill=GREEN,outline=WHITE,width=1)
    else:
        d.ellipse((x-r,y-r,x+r,y+r),fill=RED,outline=WHITE,width=1)
def plot(im,box,name,seed,level,rs,*,points=True,labels=False,crop=5.7):
    d=ImageDraw.Draw(im);l,t,r,b=box;w,h=r-l,b-t
    scale=min(w/(2*crop),h/(2*crop));cx=(l+r)/2;cy=(t+b)/2
    def coord(x,y):return (cx+x*scale,cy-y*scale)
    a=c.surface(name,seed,level);arr=Image.fromarray(colors(a).transpose(1,0,2)[::-1].copy())
    siz=round(12*scale);arr=arr.resize((siz,siz),resample=Image.Resampling.NEAREST)
    tile=Image.new('RGB',(w,h),VOID);tile.paste(arr,(round(cx-l-siz/2),round(cy-t-siz/2)))
    im.paste(tile,(l,t))
    if labels and name=='islands':
        for n,(x,y,z,hx,_,_) in enumerate(c.islands(seed,level)):
            u,v=coord(x,y+.59);txt(d,(u,v),f'I{n}  {z:.2f} m',16,INK,'mm')
    if points:
        for side in range(2):
            for x,y,z in c.patchset(c.number(name),seed,level,1,side).xyz:
                marker(d,coord(x,y),'G' if side else 'S',4 if labels else 3)
    for i,row in enumerate(rs):
        u=coord(*row['start'][:2]);v=coord(*row['goal'][:2])
        arrow(d,u,v,BLUE,3,dash=True);marker(d,u,'S',8 if labels else 5);marker(d,v,'G',8 if labels else 5)
        if labels:
            for label,pt in [(f"S{row['start_index']}",u),(f"G{row['goal_index']}",v)]:
                x,y=pt;bw=d.textbbox((0,0),label,font=font(15))[2]+8
                d.rounded_rectangle((x+7,y-18,x+7+bw,y+1),radius=3,fill=WHITE)
                txt(d,(x+11,y-16),label,15,INK)
    x,y=coord(-crop+.3,-crop+.24)
    if l<=x<r and t<=y<b:
        d.line([(x,y),(x+scale,y)],WHITE,width=3);txt(d,(x+scale/2,y-7),'1 m',12,WHITE,'mb')
    return coord
def profile(d,box,xyz,name,seed,level,title):
    l,t,r,b=box;p,q=map(np.asarray,xyz)
    ts=np.linspace(0.,1.,420);xy=p[None,:2]+ts[:,None]*(q-p)[None,:2]
    hh=c.surface_at(name,seed,level,xy[:,0],xy[:,1]);zlo=min(-.2,float(hh.min())-.12);zhi=max(.5,float(hh.max())+.28)
    def at(tx,z):return (l+tx*(r-l),b-(z-zlo)/(zhi-zlo)*(b-t))
    trace=[at(tx,z) for tx,z in zip(ts,hh)]
    d.rectangle((l,t,r,b),fill=(237,239,234),outline=(209,218,213))
    d.polygon([(l,b),*trace,(r,b)],fill=(159,183,161))
    d.line(trace,fill=MUTED,width=3)
    for z in sorted(set((round(float(hh.min()),2),round(float(hh.max()),2)))):
        txt(d,(l-7,at(0,z)[1]),f'{z:.2f}',13,MUTED,'rm')
    marker(d,at(0,p[2]),'S',6);marker(d,at(1,q[2]),'G',6)
    txt(d,(l,t-12),title,16,INK,'lb')
    txt(d,(l,b+6),'0 m',13,MUTED);txt(d,(r,b+6),f'{np.linalg.norm(q[:2]-p[:2]):.2f} m',13,MUTED,'ra')
def metric_block(d,x,y,title,route):
    txt(d,(x,y),title,23,INK);txt(d,(x,y+32),f"Task {route['id']}   |   S{route['start_index']} to G{route['goal_index']}",17,BLUE)
    for i,role in enumerate(('start','goal')):
        arr=route[role];txt(d,(x,y+65+25*i),f"{role.capitalize():5}    ({arr[0]:+.2f}, {arr[1]:+.2f})      z = {arr[2]:+.2f} m",17,INK)
    txt(d,(x,y+120),f"Horizontal {route['horizontal_m']:.2f} m     Height change {route['delta_z_m']:+.2f} m",17,MUTED)
def details(root,out,routes):
    islands=[r for r in routes if r['terrain']=='islands']
    # Two adjacent maps avoid hiding one pair's labels underneath the other.
    im=Image.new('RGB',(1450,970),PAPER);d=ImageDraw.Draw(im)
    txt(d,(46,28),'Floating islands   /   held-out task coordinates',34)
    txt(d,(46,79),'Layout 0, variant 4  |  All 20 starts and 20 goals lie on island surfaces. Surface elevations are local choices.',20,MUTED)
    for i,row in enumerate(islands[:2]):
        x=46+704*i
        plot(im,(x,130,x+660,624),'islands',0,4,[row],labels=True,crop=3.65)
        metric_block(d,x,647,'Adjacent level islands' if i==0 else 'Low island to central pillar',row)
        profile(d,(x+47,837,x+635,913),(row['start'],row['goal']),'islands',0,4,'Surface under the straight start-to-goal segment')
    im.save(out/'islands_topview.png')
    for name in ('stairs','maze'):
        rr=[r for r in routes if r['terrain']==name]
        im=Image.new('RGB',(1260,830),PAPER);d=ImageDraw.Draw(im)
        txt(d,(45,28),f'{TITLES[name]}   /   mesh, patches and reference pair',32)
        txt(d,(45,78),'Blue dashes link task endpoints. They do not prescribe a path to the policy.',19,MUTED)
        row=rr[0];plot(im,(45,125,695,718),name,0,4,[row],labels=True,crop=5.4)
        metric_block(d,733,135,'Upstairs' if name=='stairs' else 'Opposite chambers',row)
        profile(d,(762,385,1202,505),(row['start'],row['goal']),name,0,4,'Height along the endpoint chord')
        if name=='stairs':
            metric_block(d,733,560,'Downstairs, second layout',rr[1])
        else:
            txt(d,(733,562),'The chord intersects a wall.',21)
            txt(d,(733,604),'Pale floor openings alternate between sides.',17,MUTED)
            txt(d,(733,637),'A traversable path must go around the walls.',17,MUTED)
        txt(d,(45,766),'Green triangles: starts      Red circles: goals      Gray-blue: void floor      Heights in metres',18,MUTED)
        im.save(out/(name+'_detail.png'))
def overview(out,routes):
    im=Image.new('RGB',(1680,1470),PAPER);d=ImageDraw.Draw(im)
    txt(d,(46,27),'ANYmal-D   /   task bank across 13 terrains',36)
    txt(d,(46,82),'Held-out bank: mesh heights, all 20 start points, all 20 goals and one frozen route per family. Layout 0, variant 4.',21,MUTED)
    for k,name in enumerate(c.TERRAIN_NAMES):
        col=k%4;row=k//4;l=42+409*col;t=136+326*row
        rr=next(v for v in routes if v['terrain']==name)
        txt(d,(l,t),TITLES[name],22);txt(d,(l+385,t+5),f"ID {rr['id']}",15,MUTED,'ra')
        plot(im,(l,t+39,l+385,t+284),name,0,4,[rr],crop=4.0)
        txt(d,(l,t+288),f"Distance {rr['horizontal_m']:.2f} m      Height change {rr['delta_z_m']:+.2f} m",15,MUTED)
    l=465;t=1150
    marker(d,(l,t),'S',8);txt(d,(l+20,t),'Start',20,INK,'lm')
    marker(d,(l+135,t),'G',8);txt(d,(l+155,t),'Goal',20,INK,'lm')
    arrow(d,(l+270,t),(l+320,t),BLUE,3,True);txt(d,(l+332,t),'Frozen pair, not a prescribed path',20,INK,'lm')
    txt(d,(l,t+64),'Surface height relative to local datum',20)
    for i,(value,color) in enumerate(BANDS):
        x=l+i*110;d.rectangle((x,t+106,x+95,t+133),fill=color);txt(d,(x,t+139),f'{value:+.2f} m',15,MUTED)
    d.rectangle((l+680,t+106,l+775,t+133),fill=VOID);txt(d,(l+680,t+139),'Void floor',15,MUTED)
    txt(d,(l,t+205),'Colors and patches come from the same 0.15 m mesh used by the simulator.',18,MUTED)
    im.save(out/'terrain_overview.png')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('bank',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);rs=json.loads((a.bank/'fixed_routes.json').read_text());overview(a.output,rs);details(a.bank,a.output,rs)
