"""Figures 1-3, Section 3. Usage: python fig_method.py OUTDIR

Revised 18.8.2026. Changes against the previous version:

F3  - Removed the second orange arrow, which ran from the edge-feature box to the
      "weighted sum over neighbours" box. The code applies edge features to the
      attention logit only (`score += edge_bias(edge_attr)`); there is no edge term
      in the value path (`msg = v[src] * alpha`). The arrow depicted a mechanism
      that does not exist.
    - Annotation changed accordingly, and now states the negative explicitly.
    - Equation labels updated for the Section 3 renumbering: attention is Eq 9
      (was 8), aggregation Eq 10 (was 9), decoder Eq 12 (was 10). Added Eq 8 to the
      projection box and Eq 11 to the residual/FFN box.
    - Decoder box now names the final LayerNorm, which Eq 12 includes
      (`decoder(self.norm(h))`) and the previous figure omitted.

F2  - Removed both in-figure titles, per the manuscript's PENDING EDITS list. The
      caption carries this text already.
"""
import sys, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np
OUT = sys.argv[1] if len(sys.argv)>1 else "."
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size':10,'font.family':'sans-serif'})

def panel(ax,tag,x=0.0,y=1.02):
    ax.text(x,y,tag,transform=ax.transAxes,fontsize=10,fontweight='bold',va='bottom',ha='left')
BLUE,GREEN,ORANGE,GREY,LGREY='#0072B2','#009E73','#D55E00','#555555','#D9D9D9'

def save(fig,name):
    fig.savefig(os.path.join(OUT,name+'.png'),dpi=300,bbox_inches='tight')
    fig.savefig(os.path.join(OUT,name+'.pdf'),bbox_inches='tight'); plt.close(fig)

# ---------------- Figure 1: rolling-origin protocol ----------------
from matplotlib.patches import Patch
fig,ax=plt.subplots(figsize=(9.0,3.6))
W=10; rows=4
for r in range(rows):
    w = 5+r
    y = rows-1-r
    ax.add_patch(Rectangle((0,y-0.28),w-0.85,0.56,fc=LGREY,ec='none'))
    ax.add_patch(Rectangle((w-0.85,y-0.28),0.85,0.56,fc=GREEN,ec='none',alpha=.55))
    ax.add_patch(Rectangle((w,y-0.28),1,0.56,fc=BLUE,ec='none'))
    ax.text(-0.3,y,f'Fold {r+1}',ha='right',va='center',fontsize=9.5)
    if r<rows-1:
        ax.annotate('',xy=(w+1.5,y-0.5),xytext=(w+1.0,y-0.5),
                    arrowprops=dict(arrowstyle='->',color=GREY,lw=1.1))
for t in range(W+1): ax.plot([t,t],[-0.62,-0.50],color='0.6',lw=.8)
ax.plot([0,W],[-0.62,-0.62],color='0.6',lw=.9)
ax.text(W/2,-0.92,'calendar week',ha='center',fontsize=10,color='black')
ax.legend(handles=[Patch(fc=LGREY,label='fitting partition'),
                   Patch(fc=GREEN,alpha=.55,label='validation tail, last 15% by time'),
                   Patch(fc=BLUE,label='test week')],
          loc='upper center',bbox_to_anchor=(0.5,1.09),ncol=3,frameon=False,fontsize=10)
ax.set_xlim(-1.5,W+0.6); ax.set_ylim(-1.15,rows-0.5); ax.axis('off')
save(fig,'F1_protocol')

# ---------------- Figure 2: causal graph construction ----------------
fig,(a1,a2)=plt.subplots(1,2,figsize=(11.6,4.1),gridspec_kw={'width_ratios':[1.3,1]})
tx=[0.5,1.5,2.3,3.3,4.2,5.2,6.2,7.3]     # last one is a FUTURE transaction
tgt=6
card=[1,1,0,1,0,1]; merch=[0,1,1,0,1,1]
a1.plot([0.15,7.7],[0,0],color='0.7',lw=1.1,zorder=1)
for i,x in enumerate(tx[:tgt]):
    if card[i]:
        a1.add_patch(FancyArrowPatch((x,0.10),(tx[tgt]-0.10,0.12),connectionstyle='arc3,rad=-0.45',
                     arrowstyle='-|>',mutation_scale=10,color=BLUE,lw=1.3,zorder=2))
    if merch[i]:
        a1.add_patch(FancyArrowPatch((x,-0.10),(tx[tgt]-0.10,-0.12),connectionstyle='arc3,rad=0.45',
                     arrowstyle='-|>',mutation_scale=10,color=ORANGE,lw=1.3,ls='--',zorder=2))
a1.add_patch(FancyArrowPatch((tx[7],-0.10),(tx[tgt]+0.10,-0.12),connectionstyle='arc3,rad=-0.45',
             arrowstyle='-|>',mutation_scale=10,color='0.72',lw=1.3,ls=':',zorder=2))
a1.scatter(tx[:tgt],[0]*tgt,s=150,facecolor='white',edgecolor='0.45',lw=1.3,zorder=4)
a1.scatter([tx[7]],[0],s=150,facecolor='white',edgecolor='0.72',lw=1.3,ls=':',zorder=4)
a1.scatter([tx[tgt]],[0],s=210,facecolor=BLUE,edgecolor=BLUE,lw=1.3,zorder=5)
mx,my=(tx[7]+tx[tgt])/2,-0.36
a1.plot([mx-0.13,mx+0.13],[my-0.09,my+0.09],color=ORANGE,lw=2.6,zorder=6)
a1.plot([mx-0.13,mx+0.13],[my+0.09,my-0.09],color=ORANGE,lw=2.6,zorder=6)
a1.text(tx[tgt],0.86,'transaction being scored',fontsize=10,color=BLUE,ha='center')
a1.annotate('',xy=(tx[tgt],0.14),xytext=(tx[tgt],0.78),arrowprops=dict(arrowstyle='->',color=BLUE,lw=.9))
a1.text(tx[7],0.30,'later\ntransaction',fontsize=10,color='black',ha='center')
a1.text(0.15,0.60,'same cardholder',fontsize=10,color=BLUE)
a1.text(0.15,-0.66,'same merchant',fontsize=10,color=ORANGE)
a1.text(mx+0.30,my,'edges from the future\nare never created',fontsize=10,color=ORANGE,va='center')
a1.annotate('',xy=(7.7,-0.95),xytext=(0.15,-0.95),arrowprops=dict(arrowstyle='->',color='0.55',lw=1))
a1.text(3.9,-1.12,'time',fontsize=10,color='black',ha='center')
a1.set_xlim(-0.1,9.6); a1.set_ylim(-1.25,1.05); a1.axis('off')
a2.set_xlim(0,10); a2.set_ylim(2.9,9.4); a2.axis('off')
ys=[8.3,7.4,6.5,5.6,4.7,3.6]
labs=['most recent prior','2nd most recent','3rd most recent','...','K-th most recent','older than the K-th']
for i,(y,l) in enumerate(zip(ys,labs)):
    keep=i<5
    a2.add_patch(Rectangle((0.5,y-0.3),5.0,0.6,fc=GREEN if keep else 'white',alpha=.30 if keep else 1,
                 ec='none' if keep else GREY,ls='-' if keep else ':',lw=1))
    a2.text(0.8,y,l,fontsize=10,va='center',color='black')
    a2.text(5.9,y,'kept' if keep else 'discarded',fontsize=10,va='center',color='black')
panel(a1,'(a)'); panel(a2,'(b)')
save(fig,'F2_graph')
print('wrote F1_protocol and F2_graph to',os.path.abspath(OUT))

# ---------------- Figure 3: architecture ----------------
fig,ax=plt.subplots(figsize=(11.8,4.6))
ax.set_xlim(0,28); ax.set_ylim(0,11); ax.axis('off')
def box(x,y,w,h,label,fc='white',ec=GREY,fs=9.5,bold=False,ls='-'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.10,rounding_size=0.16',
                 fc=fc,ec=ec,lw=1.3,linestyle=ls))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=fs,
            fontweight='bold' if bold else 'normal',linespacing=1.4)
def arrow(x1,y1,x2,y2,c=GREY,ls='-'):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=12,
                 color=c,lw=1.3,linestyle=ls,zorder=1))

box(0.4,6.55,3.6,1.7,'node features\n$x_i$',fc='#EAF3FA',ec=BLUE)
box(4.9,6.55,3.3,1.7,'linear projection\nand GELU\nEq. 8',ec=BLUE,fs=9)
arrow(4.1,7.4,4.8,7.4,BLUE)
ax.add_patch(FancyBboxPatch((9.3,3.3),9.6,6.2,boxstyle='round,pad=0.12,rounding_size=0.2',
             fc='#F7F7F7',ec='0.75',lw=1.2,linestyle='--'))
ax.text(14.1,9.85,'message-passing block, repeated $L$ times',ha='center',fontsize=9,color='black')
box(9.70,7.2,4.30,1.7,'temporal relational\nattention\nEq. 9',fc='#EAF3FA',ec=BLUE,fs=9)
box(14.35,7.2,4.30,1.7,'weighted sum over\nneighbours\nEq. 10',fc='#EAF3FA',ec=BLUE,fs=9)
box(11.60,4.0,5.00,1.7,'pre-norm residual\nand feed-forward\nEq. 11',ec=GREY,fs=9)
arrow(8.2,7.4,9.6,8.05,BLUE)
arrow(14.10,8.05,14.45,8.05,BLUE)
arrow(16.4,7.1,15.5,5.8,GREY)
arrow(12.7,5.8,11.9,7.1,GREY)
box(0.4,1.2,3.6,1.6,'5-dim edge\nfeatures  $e_{ji}$',fc='#FDEEE4',ec=ORANGE)
ax.text(2.2,0.85,'elapsed time, relation,\namount ratio',ha='center',va='top',fontsize=9,color='black')
ax.add_patch(FancyArrowPatch((4.1,2.3),(11.8,7.1),connectionstyle='arc3,rad=-0.22',
             arrowstyle='-|>',mutation_scale=12,color=ORANGE,lw=1.3,linestyle=(0,(4,2)),zorder=1))
box(20.0,6.55,3.5,1.7,'layer norm and\nMLP decoder\nEq. 12',ec=GREY,fs=9)
arrow(19.0,7.4,19.9,7.4)
box(24.3,6.55,3.2,1.7,'fraud score\n$s_i$',fc='#E6F5F0',ec=GREEN)
arrow(23.5,7.4,24.2,7.4,GREEN)
ax.text(25.9,5.90,'ranked, top $k$\nsent for review',ha='center',va='top',fontsize=9,color='black')
save(fig,'F3_architecture')
print('wrote F1_protocol, F2_graph and F3_architecture to',os.path.abspath(OUT))
