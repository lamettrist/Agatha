"""Plot actual paired benchmark results; no synthetic or placeholder metrics."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

results=json.loads(Path('models/ab_results.json').read_text())['models']
suites=['layout','toss','noclip','toss_noclip','stacks','more_objects']
labels=['Layouts','Bottle toss','Ghost table','Toss + ghost','Stacks','9 objects']
colors={'mlp_multistep':'#237b66','graph_impulse':'#8061b0','physics':'#8f9aa3'}
fig,axes=plt.subplots(1,3,figsize=(15,4.3),gridspec_kw={'width_ratios':[1.6,1.1,.8]})
x=np.arange(len(suites))
for offset,kind in [(-.18,'mlp_multistep'),(.18,'graph_impulse')]:
 values=[results[kind][s]['mean_position_error_m']['2.0'] for s in suites]
 axes[0].bar(x+offset,values,.34,label='MLP' if kind=='mlp_multistep' else 'Graph',color=colors[kind])
axes[0].set_xticks(x,labels,rotation=28,ha='right');axes[0].set_ylabel('Mean position error at 2 seconds (m)');axes[0].set_title('Same held-out scenes and PyBullet reference');axes[0].legend(frameon=False)
solid=['layout','toss','stacks','more_objects'];x=np.arange(len(solid))
for offset,kind in [(-.24,'physics'),(0,'mlp_multistep'),(.24,'graph_impulse')]:
 axes[1].bar(x+offset,[results[kind][s]['passed_through_solid_table_objects'] for s in solid],.23,label=kind,color=colors[kind])
axes[1].set_xticks(x,['Layouts','Toss','Stacks','9 objects'],rotation=28,ha='right');axes[1].set_ylabel('Objects crossing through a solid table');axes[1].set_title('Clipping is still an observed failure')
for i,kind in enumerate(['mlp_multistep','graph_impulse']):
 values=[results[kind][s]['median_rollout_ms'] for s in suites]
 axes[2].bar(i,float(np.mean(values)),color=colors[kind]);axes[2].text(i,float(np.mean(values))+1,f'{np.mean(values):.0f} ms',ha='center',fontsize=10)
axes[2].set_xticks([0,1],['MLP','Graph']);axes[2].set_ylabel('Mean suite median latency (ms)');axes[2].set_title('2 s rollout / four CPU threads');axes[2].set_ylim(0,130)
for ax in axes:
 ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('Agatha world-model A/B — 6,000 shared episodes (5,400 train), 250 paired test scenes',fontsize=13,y=1.02)
fig.tight_layout();Path('docs').mkdir(exist_ok=True);fig.savefig('docs/world-model-ab.png',dpi=160,bbox_inches='tight');fig.savefig('docs/world-model-ab.svg',bbox_inches='tight')
print('Saved docs/world-model-ab.png and .svg from measured results')
