from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

COMPONENT_COLS = [
    'score_stream_core_usable',
    'score_seed_trait_usable',
    'score_stream_seed_trait_usable',
    'score_cadence',
    'score_member_role',
    'score_exact_stream_core_member',
]
MAJOR_COMPONENTS = [
    'score_stream_core_usable',
    'score_seed_trait_usable',
    'score_stream_seed_trait_usable',
    'score_cadence',
    'score_exact_stream_core_member',
]

KEY_TRAIT_COLS = ['seed_parity','seed_highlow','seed_structure','seed_parity_count','seed_sum_bucket']

def _num(s):
    return pd.to_numeric(s, errors='coerce').fillna(0)

def profile_inventory(profile_dir) -> pd.DataFrame:
    rows=[]
    p=Path(profile_dir)
    for fp in sorted(p.glob('*.csv')):
        try:
            df=pd.read_csv(fp, dtype=str)
            cols=list(df.columns)
            score_cols=[c for c in cols if any(tok in c.lower() for tok in ['score','support','confidence','hit','lift','precision','count'])]
            key_cols=[c for c in cols if c in ['StreamKey','core_str','member_str','target_core','candidate_member','trait_name','trait_value','SameCoreGapBucket','stream','PLAY_DATE','HISTORY_THROUGH']]
            rows.append({
                'file':fp.name,
                'rows':len(df),
                'columns':len(cols),
                'key_columns':' | '.join(key_cols),
                'score_support_columns':' | '.join(score_cols[:30]),
                'has_StreamKey':'StreamKey' in cols,
                'has_core_str':'core_str' in cols,
                'has_member_str':'member_str' in cols,
                'has_target_core':'target_core' in cols,
                'has_support_column':'support' in cols or 'total_support' in cols,
            })
        except Exception as e:
            rows.append({'file':fp.name,'rows':np.nan,'columns':np.nan,'error':str(e)})
    return pd.DataFrame(rows)

def support_summary(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows=[]
    d=df.copy()
    for col in COMPONENT_COLS + ['major_support_count','all_support_count']:
        if col in d.columns:
            s=_num(d[col])
            rows.append({
                'table':label,'field':col,'present':True,'rows':len(d),
                'nonzero_rows':int(s.gt(0).sum()),
                'pct_nonzero':round(float(s.gt(0).mean()*100),2) if len(s) else 0,
                'min':float(s.min()) if len(s) else np.nan,
                'mean':float(s.mean()) if len(s) else np.nan,
                'max':float(s.max()) if len(s) else np.nan,
                'distinct_values':int(s.nunique()) if len(s) else 0,
            })
        else:
            rows.append({'table':label,'field':col,'present':False,'rows':len(d),'nonzero_rows':0,'pct_nonzero':0})
    return pd.DataFrame(rows)

def recompute_support_counts(df: pd.DataFrame) -> pd.DataFrame:
    d=df.copy()
    for col in COMPONENT_COLS:
        if col not in d.columns:
            d[col]=0
    d['debug_recomputed_major_support_count']=sum(_num(d[c]).gt(0).astype(int) for c in MAJOR_COMPONENTS)
    d['debug_recomputed_all_support_count']=sum(_num(d[c]).gt(0).astype(int) for c in COMPONENT_COLS)
    d['support_count_mismatch_major']=False
    d['support_count_mismatch_all']=False
    if 'major_support_count' in d.columns:
        d['support_count_mismatch_major']=_num(d['major_support_count']).astype(int).ne(d['debug_recomputed_major_support_count'].astype(int))
    if 'all_support_count' in d.columns:
        d['support_count_mismatch_all']=_num(d['all_support_count']).astype(int).ne(d['debug_recomputed_all_support_count'].astype(int))
    return d

def join_audit_rows(step2: pd.DataFrame, limit: int|None=None) -> pd.DataFrame:
    d=recompute_support_counts(step2)
    rows=[]
    take=d if limit is None else d.head(limit)
    for _,r in take.iterrows():
        trait_hits=[]
        for c in KEY_TRAIT_COLS:
            if c in r.index and pd.notna(r.get(c,'')) and str(r.get(c,''))!='':
                trait_hits.append(f'{c}={r.get(c)}')
        rec={
            'play_date':r.get('play_date',''),
            'stream':r.get('stream',''),
            'seed':r.get('seed',''),
            'core':str(r.get('core','')).zfill(3),
            'member':str(r.get('member','')).zfill(4),
            'same_core_gap_bucket':r.get('same_core_gap_bucket',''),
            'seed_trait_values':' | '.join(trait_hits),
            'major_support_count_saved':r.get('major_support_count',np.nan),
            'all_support_count_saved':r.get('all_support_count',np.nan),
            'major_support_count_recomputed':r.get('debug_recomputed_major_support_count',np.nan),
            'all_support_count_recomputed':r.get('debug_recomputed_all_support_count',np.nan),
            'mismatch_major':r.get('support_count_mismatch_major',False),
            'mismatch_all':r.get('support_count_mismatch_all',False),
        }
        for c in COMPONENT_COLS:
            rec[c]=r.get(c,0)
            rec[c+'_fired']=float(pd.to_numeric(pd.Series([r.get(c,0)]), errors='coerce').fillna(0).iloc[0])>0
        rows.append(rec)
    return pd.DataFrame(rows)

def component_summary_by_core_stream(step2: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    d=recompute_support_counts(step2)
    for c in COMPONENT_COLS + ['major_support_count','all_support_count','debug_recomputed_major_support_count','debug_recomputed_all_support_count']:
        if c in d.columns: d[c]=_num(d[c])
    agg={}
    for c in COMPONENT_COLS:
        agg[c+'_fired_rows']=(c, lambda x: int(_num(x).gt(0).sum()))
        agg[c+'_max']=(c,'max')
    agg.update({
        'rows':('member','size'),
        'saved_major_max':('major_support_count','max'),
        'recomputed_major_max':('debug_recomputed_major_support_count','max'),
        'saved_all_max':('all_support_count','max'),
        'recomputed_all_max':('debug_recomputed_all_support_count','max'),
        'mismatch_major_rows':('support_count_mismatch_major', lambda x: int(pd.Series(x).fillna(False).sum())),
    })
    by_core=d.groupby(['play_date','core'], as_index=False).agg(**agg) if {'play_date','core'}.issubset(d.columns) else pd.DataFrame()
    by_stream=d.groupby(['play_date','stream'], as_index=False).agg(**agg) if {'play_date','stream'}.issubset(d.columns) else pd.DataFrame()
    return by_core, by_stream

def diagnosis(profile_dir, full: pd.DataFrame, step2_before: pd.DataFrame|None, step2_after: pd.DataFrame) -> pd.DataFrame:
    inv=profile_inventory(profile_dir)
    full_s=support_summary(full,'full_step1')
    step2_s=support_summary(step2_after,'step2_after_transition')
    diag=[]
    profile_files=len(inv)
    profile_rows=int(pd.to_numeric(inv.get('rows', pd.Series(dtype=float)), errors='coerce').fillna(0).sum()) if not inv.empty else 0
    full_major_nonzero=int(full_s.loc[full_s.field.eq('major_support_count'),'nonzero_rows'].iloc[0]) if not full_s.empty and full_s.field.eq('major_support_count').any() else 0
    step2_major_nonzero=int(step2_s.loc[step2_s.field.eq('major_support_count'),'nonzero_rows'].iloc[0]) if not step2_s.empty and step2_s.field.eq('major_support_count').any() else 0
    full_component_nonzero=int(full_s[full_s.field.isin(COMPONENT_COLS)]['nonzero_rows'].sum()) if not full_s.empty else 0
    step2_component_nonzero=int(step2_s[step2_s.field.isin(COMPONENT_COLS)]['nonzero_rows'].sum()) if not step2_s.empty else 0
    diag.append({'check':'profile_files_loaded','value':profile_files,'status':'OK' if profile_files>0 else 'FAIL','note':'Number of CSVs in profiles/.'})
    diag.append({'check':'profile_total_rows','value':profile_rows,'status':'OK' if profile_rows>0 else 'FAIL','note':'Total profile/rule rows available.'})
    diag.append({'check':'full_step1_major_support_nonzero_rows','value':full_major_nonzero,'status':'OK' if full_major_nonzero>0 else 'WARN','note':'If zero, support joins failed before Step 2.'})
    diag.append({'check':'step2_major_support_nonzero_rows','value':step2_major_nonzero,'status':'OK' if step2_major_nonzero>0 else 'FAIL','note':'If zero but Step1 nonzero, support was lost during Step2 handoff/scope/export.'})
    diag.append({'check':'full_step1_component_nonzero_total','value':full_component_nonzero,'status':'OK' if full_component_nonzero>0 else 'WARN','note':'Nonzero score components in Step1.'})
    diag.append({'check':'step2_component_nonzero_total','value':step2_component_nonzero,'status':'OK' if step2_component_nonzero>0 else 'FAIL','note':'Nonzero score components still present in Step2.'})
    # recompute mismatch
    if step2_after is not None and not step2_after.empty:
        rec=recompute_support_counts(step2_after)
        mism_major=int(rec['support_count_mismatch_major'].sum()) if 'support_count_mismatch_major' in rec else 0
        mism_all=int(rec['support_count_mismatch_all'].sum()) if 'support_count_mismatch_all' in rec else 0
        diag.append({'check':'major_support_saved_vs_recomputed_mismatch_rows','value':mism_major,'status':'FAIL' if mism_major>0 else 'OK','note':'Mismatch means saved major_support_count is wrong even though component scores are present.'})
        diag.append({'check':'all_support_saved_vs_recomputed_mismatch_rows','value':mism_all,'status':'FAIL' if mism_all>0 else 'OK','note':'Mismatch means saved all_support_count is wrong even though component scores are present.'})
    return pd.DataFrame(diag)

def write_support_debug_outputs(out_dir, profile_dir, play_date, full, step2_before, step2_after):
    out=Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    suffix=str(play_date)
    inv=profile_inventory(profile_dir)
    inv.to_csv(out/'SUPPORT_00_PROFILE_INVENTORY.csv', index=False)
    sumdf=pd.concat([
        support_summary(full,'full_step1'),
        support_summary(step2_before,'step2_before_transition') if step2_before is not None else pd.DataFrame(),
        support_summary(step2_after,'step2_after_transition')
    ], ignore_index=True)
    sumdf.to_csv(out/f'SUPPORT_01_SIGNAL_SUMMARY_{suffix}.csv', index=False)
    rec=recompute_support_counts(step2_after)
    audit_cols=['play_date','stream','seed','core','member'] + COMPONENT_COLS + ['major_support_count','all_support_count','debug_recomputed_major_support_count','debug_recomputed_all_support_count','support_count_mismatch_major','support_count_mismatch_all']
    rec[[c for c in audit_cols if c in rec.columns]].to_csv(out/f'SUPPORT_02_RECOMPUTED_COUNTS_{suffix}.csv', index=False)
    join_audit_rows(step2_after).to_csv(out/f'SUPPORT_03_JOIN_AUDIT_ROWS_{suffix}.csv', index=False)
    by_core, by_stream=component_summary_by_core_stream(step2_after)
    by_core.to_csv(out/f'SUPPORT_04_BY_CORE_{suffix}.csv', index=False)
    by_stream.to_csv(out/f'SUPPORT_05_BY_STREAM_{suffix}.csv', index=False)
    diagnosis(profile_dir, full, step2_before, step2_after).to_csv(out/f'SUPPORT_99_DIAGNOSIS_{suffix}.csv', index=False)
