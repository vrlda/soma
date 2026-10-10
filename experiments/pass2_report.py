"""Aggregate pass-2 raw records, registered gates, and one scientific figure."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]


def rows(name):
    p=ROOT/'experiments/results'/name
    return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []


def describe(values):
    if not values:return None
    return dict(mean=statistics.mean(values),min=min(values),max=max(values),n=len(values))


def groups(records):
    out={}
    for variant in sorted({r['variant'] for r in records}):
        rr=[r for r in records if r['variant']==variant]
        out[variant]=dict(runs=len(rr),complete=sum(r.get('complete',True) for r in rr),
           withheld=describe([r['metrics']['accuracy'] for r in rr]),
           seen=describe([r['seen_probe']['accuracy'] for r in rr]),
           initial=describe([r['initial_transfer']['accuracy'] for r in rr]),
           frozen_holdout=describe([r['frozen_hidden_holdout']['accuracy'] for r in rr
                                   if r.get('frozen_hidden_holdout',{}).get('accuracy') is not None]),
           units=describe([r['brain']['hidden'] for r in rr]))
    return out


def gates(records,selection):
    g=groups(records);required=selection['required_controls'];n=len(selection['seeds'])
    candidate=g.get('predictive');criteria={}
    criteria['all_required_runs_complete']=all(v in g and g[v]['runs']==n and g[v]['complete']==n
                                               for v in ['predictive']+required)
    if not criteria['all_required_runs_complete']:return criteria
    eps=1e-12;limits=selection['gates']
    criteria['withheld_mean']=candidate['withheld']['mean']+eps>=limits['withheld_mean']
    criteria['seen_mean']=candidate['seen']['mean']+eps>=limits['seen_mean']
    criteria['individual_seeds']=sum(r['metrics']['accuracy']+eps>=limits['individual_min']
                                    for r in records if r['variant']=='predictive')>=limits['individual_passing_count']
    for v in required:
        criteria['withheld_advantage_'+v]=candidate['withheld']['mean']-g[v]['withheld']['mean']+eps>=limits['required_control_advantage']
        criteria['initial_advantage_'+v]=candidate['initial']['mean']-g[v]['initial']['mean']+eps>=limits['initial_probe_advantage']
    criteria['paired_streams']=all(len({r['stream_hash'] for r in records if r['seed']==seed})==1
                                   for seed in selection['seeds'])
    criteria['exact_capacity']=all(len({r['brain']['synapses'] for r in records
         if r['seed']==seed and r['variant'] in ['predictive','frozen','shuffled','matched_capacity','hebbian_capacity']})==1
         for seed in selection['seeds'])
    criteria['structural_schedules']=all(len({r['schedule_hash'] for r in records
         if r['seed']==seed and r['variant'] in ['predictive','frozen','shuffled','matched_capacity','hebbian_capacity']})==1
         for seed in selection['seeds'])
    return criteria


def uncertainty_summary(records):
    out={}
    for condition in sorted({r['condition'] for r in records}):
        rr=[r for r in records if r['condition']==condition]
        out[condition]=dict(accuracy=describe([r['metrics']['accuracy'] for r in rr]),
                confidence=describe([r['mean_confidence'] for r in rr]),errors=sum(r['errors'] for r in rr),
                auc={key:describe([r['auc'][key] for r in rr if r['auc'][key] is not None])
                     for key in ('view_residual','query_residual','uncertainty')})
    return out


def figure(records,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    g=groups(records)
    variants=['predictive','frozen','shuffled','matched_capacity','hebbian_capacity','linear_memory','competitive']
    if not all(v in g and g[v]['runs']==10 and g[v]['complete']==10 for v in variants):return
    labels=['Local\nprediction','Frozen\nprediction','Shuffled\nsignal','Matched\nrandom','Matched\nOja','Linear\nmemory','Original\npool']
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained',width_ratios=(1.6,1))
    means=[g[v]['withheld']['mean'] for v in variants]
    lower=[x-g[v]['withheld']['min'] for x,v in zip(means,variants)]
    upper=[g[v]['withheld']['max']-x for x,v in zip(means,variants)]
    colors=['#187d76']+['#b18446']*4+['#367ca1','#787878']
    axes[0].bar(range(len(variants)),means,color=colors,yerr=[lower,upper],capsize=3)
    axes[0].set_xticks(range(len(variants)),labels,fontsize=9)
    axes[0].axhline(.5,color='#777777',linestyle=':',label='Chance')
    axes[0].set(title='Unseen combinations — 10 confirmation seeds',ylabel='Accuracy',ylim=(0,1.07))
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(fontsize=9)
    gg=rows('pass2_predictive_rank_diagnostic.jsonl')
    names=['predictive','frozen','shuffled','hebbian_capacity']
    for shift,field,label,color in ((-.17,'signal_fraction','Weight energy in signal span','#187d76'),
                                   (.17,'signal_rank_ratio','Second / first signal eigenvalue','#b18446')):
        values=[statistics.mean(r[field] for r in gg if r['variant']==v) for v in names]
        axes[1].bar([i+shift for i in range(4)],values,width=.34,label=label,color=color)
    axes[1].set_xticks(range(4),['Prediction','Frozen','Shuffled','Oja'],fontsize=9)
    axes[1].set(title='Learned geometry — 5 exploratory seeds',ylabel='Fraction / ratio',ylim=(0,1.07))
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    fig.suptitle('SOMA pass 2: local predictive features on a linear composition task')
    fig.savefig(output/'pass2_predictive.png',dpi=180);plt.close(fig)


def main():
    output=ROOT/'experiments/report';output.mkdir(exist_ok=True)
    records=rows('pass2_composition_confirmation.jsonl')
    selection=json.loads((ROOT/'experiments/SELECTED_PREDICTIVE.json').read_text())
    manifest=[];suites=Counter()
    for p in sorted((ROOT/'experiments/results').glob('pass2_*.jsonl')):
        data=p.read_bytes();rr=[json.loads(s) for s in data.decode().splitlines()]
        suites.update(r['suite'] for r in rr)
        manifest.append(dict(file=p.name,runs=len(rr),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    criteria=gates(records,selection)
    result=dict(runs=sum(suites.values()),suites=dict(suites),files=manifest,
                confirmation=groups(records),gates=criteria,
                registered_gates_pass=bool(criteria) and all(criteria.values()),
                uncertainty=uncertainty_summary(rows('pass2_predictive_uncertainty.jsonl')))
    (output/'PASS2.json').write_text(json.dumps(result,indent=2)+'\n')
    def pct(x):return '—' if x is None else f"{100*x['mean']:.2f}% [{100*x['min']:.2f}, {100*x['max']:.2f}]"
    lines=['# Pass 2 measurements','',
      'Generated by `python -B -m experiments.pass2_report`. Ranges are observed seed ranges, not confidence intervals.',
      '',f"Registered confirmation gates pass: **{result['registered_gates_pass']}**.",'',
      '| Variant | Complete | Held out | Seen | Original 4 probes | Frozen-hidden 160 probes | Predictor units |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for v,g in result['confirmation'].items():
        lines.append(f"| {v} | {g['complete']}/{g['runs']} | {pct(g['withheld'])} | {pct(g['seen'])} | {pct(g['initial'])} | {pct(g['frozen_holdout'])} | {g['units']['mean']:.1f} |")
    lines+=['','## Registered checks','']+[f"- {name}: {passed}" for name,passed in criteria.items()]
    lines+=['','## Unlabelled uncertainty probes','',
      'AUC values are means of defined within-condition, within-seed AUCs; labels never reach the brain.',
      '', '| Condition | Accuracy | Confidence | Errors | View residual AUC | Query residual AUC | Entropy AUC |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for name,g in result['uncertainty'].items():
        def number(x):return '—' if x is None else f"{x['mean']:.3f}"
        lines.append(f"| {name} | {pct(g['accuracy'])} | {pct(g['confidence'])} | {g['errors']} | {number(g['auc']['view_residual'])} | {number(g['auc']['query_residual'])} | {number(g['auc']['uncertainty'])} |")
    lines+=['','## Raw-file manifest','', '| File | Records | SHA-256 |','|---|---:|---|']
    lines+=[f"| {r['file']} | {r['runs']} | {r['sha256']} |" for r in manifest]
    (output/'PASS2.md').write_text('\n'.join(lines)+'\n')
    figure(records,output)
    print(json.dumps(dict(runs=result['runs'],gates=criteria,registered_gates_pass=result['registered_gates_pass']),indent=2))

if __name__=='__main__':main()
