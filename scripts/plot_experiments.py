"""Render publication-ready PNG/SVG figures from verified summary JSON."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();data=json.loads(args.results.read_text(encoding='utf-8'));groups=data['groups']
    args.output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    names=list(groups);colors=['#2176ae','#d65c45'];x=np.arange(len(names))
    def save(fig,name):
        fig.savefig(args.output/(name+'.png'),dpi=220,bbox_inches='tight');fig.savefig(args.output/(name+'.svg'),bbox_inches='tight');plt.close(fig)
        svg=args.output/(name+'.svg')
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,metric,title in zip(axes[:2],('map50','map50_95'),('Test mAP50','Test mAP50-95')):
        for i,name in enumerate(names):
            s=groups[name]['aggregate'][metric]
            if s['n']:
                ax.bar(i,s['mean'],color=colors[i],width=.55,yerr=s['std'] if s['n']>1 else None,capsize=5)
                ax.text(i,s['mean']+.035,f"n={s['n']}",ha='center')
            else:ax.text(i,.1,'Not completed',ha='center')
        ax.set(xticks=x,xticklabels=names,ylim=(0,1.08),title=title,ylabel='Score')
    for i,name in enumerate(names):
        values=[r['training_seconds']/60 for r in groups[name]['completed']]
        if values:
            axes[2].bar(i,np.mean(values),color=colors[i],width=.55,yerr=np.std(values,ddof=1) if len(values)>1 else None,capsize=5)
            axes[2].scatter([i]*len(values),values,color='#252525',s=15,zorder=3)
        else:axes[2].text(i,.1,'Not completed',ha='center')
    axes[2].set(xticks=x,xticklabels=names,title='Training time',ylabel='Minutes per seed')
    fig.suptitle(f"E1 original training vs E2 train-only tiles — {data['status']}",y=1.04)
    fig.text(.5,-.04,'Error bars: sample SD across training seeds; shared original-image test set. Equal epoch ceiling, unequal compute.',ha='center',fontsize=9)
    fig.tight_layout();save(fig,'comparison')
    labels=groups['E1']['protocol']['config']['class_map'];fig,axes=plt.subplots(1,2,figsize=(10,4));cx=np.arange(len(labels));width=.35
    for ax,metric,title in zip(axes,('recall','map50_95'),('Per-class test recall','Per-class test mAP50-95')):
        for i,name in enumerate(names):
            values=[groups[name]['per_class'][cid][metric] for cid in labels]
            if any(v['n'] for v in values):ax.bar(cx+(i-.5)*width,[v['mean'] or 0 for v in values],width,label=name,color=colors[i],yerr=[v['std'] or 0 for v in values],capsize=3)
        ax.set(xticks=cx,xticklabels=list(labels.values()),ylim=(0,1.1),title=title,ylabel='Score');ax.legend()
    fig.tight_layout();save(fig,'per-class')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,name in zip(axes,names):
        row=next((r for r in groups[name]['completed'] if r['seed']==42),None)
        if row is None:ax.text(.5,.5,'Seed 42 not completed',ha='center');ax.set_axis_off();continue
        matrix=np.array(row['metrics']['confusion_matrix']);labels=row['metrics']['confusion_matrix_labels']
        image=ax.imshow(matrix,cmap='Blues');fig.colorbar(image,ax=ax,fraction=.046)
        for (r,c),v in np.ndenumerate(matrix):ax.text(c,r,str(int(v)),ha='center',va='center',color='white' if v>matrix.max()/2 else 'black',fontsize=8)
        ax.set(xticks=range(len(labels)),xticklabels=labels,yticks=range(len(labels)),yticklabels=labels,xlabel='True',ylabel='Predicted',title=name+' seed 42: test counts')
        ax.tick_params(axis='x',rotation=25)
    fig.tight_layout();save(fig,'confusion-seed42')


if __name__=='__main__':main()
