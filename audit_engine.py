from __future__ import annotations
import re, zipfile, time
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
import numpy as np

import daily_ladder_engine as eng


# ---- embedded support_debug module (single-file Streamlit Cloud safety) ----
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
    search_dirs=[]
    for d in [p, Path.cwd(), Path(__file__).resolve().parent, Path(__file__).resolve().parent/'profiles']:
        try:
            rd=d.resolve()
        except Exception:
            rd=d
        if d.exists() and rd not in search_dirs:
            search_dirs.append(rd)
    files=[]
    seen=set()
    for d in search_dirs:
        for fp in sorted(Path(d).glob('*.csv')):
            if fp.name.startswith('SUPPORT_') or fp.name.startswith('0') or fp.name.startswith('STEP2_'):
                continue
            key=fp.name.lower()
            if key not in seen:
                seen.add(key); files.append(fp)
    for fp in files:
        try:
            df=pd.read_csv(fp, dtype=str)
            cols=list(df.columns)
            score_cols=[c for c in cols if any(tok in c.lower() for tok in ['score','support','confidence','hit','lift','precision','count'])]
            key_cols=[c for c in cols if c in ['StreamKey','core_str','member_str','target_core','candidate_member','trait_name','trait_value','SameCoreGapBucket','stream','PLAY_DATE','HISTORY_THROUGH']]
            rows.append({
                'file':fp.name,
                'source_dir':str(fp.parent),
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
    diag.append({'check':'profile_files_loaded','value':profile_files,'status':'OK' if profile_files>0 else 'FAIL','note':'Number of profile CSVs found in profiles/ OR repo root fallback.'})
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

class _EmbeddedSupportDebug:
    write_support_debug_outputs = staticmethod(write_support_debug_outputs)

sd = _EmbeddedSupportDebug()
# ---- end embedded support_debug ----

BUILD_ID = "MEMBER_LOCATION_AUDITOR_V5_RANK_LIFT"
BUILD_LABEL = "Member Location Auditor V5 — rank-lift audit + support debug"
WATCHED8 = set(getattr(eng, 'WATCHED8', {'027','067','138','145','389','457','567','679'}))

BUCKET_BASIS_OPTIONS = [
    'final_x15_positive',
    'major_ge3',
    'major_ge4',
    'good_transition_ge1',
    'good_ge1_no_bad',
    'major_ge3_and_good_ge1',
    'major_ge4_and_good_ge1',
    'profile_score_positive',
    'all_step2_rows_cartesian_reference',
]
STEP2_SCOPE_OPTIONS = ['watched8_all_members','watched8_positive_support','full120_all_members','legacy_q2_balanced']

CORE_FILTERS = [
    'CORE_all','CORE_above_mean','CORE_at_mean','CORE_below_mean','CORE_at_or_above_mean','CORE_at_or_below_mean',
    'CORE_above_median','CORE_at_median','CORE_below_median','CORE_at_or_above_median','CORE_at_or_below_median','CORE_is_max','CORE_is_min'
]
STREAM_FILTERS = [
    'STREAM_all','STREAM_above_mean','STREAM_at_mean','STREAM_below_mean','STREAM_at_or_above_mean','STREAM_at_or_below_mean',
    'STREAM_above_median','STREAM_at_median','STREAM_below_median','STREAM_at_or_above_median','STREAM_at_or_below_median','STREAM_is_max','STREAM_is_min'
]
QUAL_FILTERS = [
    'all_rows','safe_no_bad','aggressive_good1_no_bad','major_ge4','major_ge4_and_aggressive','major_ge3_and_good_ge1', 'major_ge4_and_good_ge1'
]


TRANSITION_FILTERS = [
    'TRANS_all',
    'TRANS_score_ge_0', 'TRANS_score_ge_0_25', 'TRANS_score_ge_0_5', 'TRANS_score_ge_1',
    'GOOD_ge1', 'GOOD_ge2', 'GOOD_ge3', 'GOOD_ge4',
    'BAD_eq0', 'BAD_le1', 'BAD_le2',
    'GOOD_ge1_BAD_le1', 'GOOD_ge1_BAD_le2', 'GOOD_ge2_BAD_le2',
    'X15_ge_0', 'X15_ge_10', 'X15_ge_15', 'X15_ge_20',
]

# Compact, targeted filters for the immediate playable-system lockdown search.
LOCKDOWN_CORE_FILTERS = [
    'CORE_all', 'CORE_at_or_above_median', 'CORE_is_max', 'CORE_at_or_above_median_OR_is_max'
]
LOCKDOWN_STREAM_FILTERS = [
    'STREAM_all', 'STREAM_at_or_above_median', 'STREAM_is_max', 'STREAM_at_or_below_mean', 'STREAM_below_mean'
]
LOCKDOWN_QUAL_FILTERS = [
    'all_rows', 'major_ge3_and_good_ge1', 'major_ge4_and_good_ge1', 'major_ge4_and_aggressive'
]
LOCKDOWN_TRANSITION_FILTERS = [
    'TRANS_all', 'TRANS_score_ge_1', 'GOOD_ge1', 'GOOD_ge2', 'GOOD_ge1_BAD_le2', 'X15_ge_10', 'X15_ge_15'
]


def normalize_date(x):
    return pd.to_datetime(x).strftime('%Y-%m-%d')


def display_stream(state, game):
    state = '' if pd.isna(state) else str(state).strip()
    game = '' if pd.isna(game) else str(game).strip()
    if state and game: return f"{state} | {game}"
    return state or game


def read_winners_any(path_or_file) -> pd.DataFrame:
    """Read winner targets from TXT/CSV. Returns play_date, state, game, stream, result, member, core."""
    if path_or_file is None:
        return pd.DataFrame(columns=['play_date','state','game','stream','result','member','core','is_watched8_core'])
    # Streamlit uploads have .name and .read; local paths are strings/Path.
    if hasattr(path_or_file, 'read'):
        name = getattr(path_or_file, 'name', 'uploaded_winners')
        data = path_or_file.read()
        if isinstance(data, bytes): text = data.decode('utf-8', errors='ignore')
        else: text = str(data)
        suffix = Path(name).suffix.lower()
    else:
        p = Path(path_or_file)
        text = p.read_text(encoding='utf-8', errors='ignore')
        suffix = p.suffix.lower()

    if suffix == '.csv':
        from io import StringIO
        df = pd.read_csv(StringIO(text), dtype=str)
        cols = {c.lower().strip(): c for c in df.columns}
        date_col = cols.get('date') or cols.get('draw_date') or cols.get('play_date')
        state_col = cols.get('state')
        game_col = cols.get('game')
        result_col = cols.get('result') or cols.get('result4') or cols.get('base4') or cols.get('winner')
        stream_col = cols.get('stream') or cols.get('streamkey') or cols.get('stream_name')
        rows=[]
        for _,r in df.iterrows():
            result = r.get(result_col, '') if result_col else ''
            m = re.search(r'(\d)[-\s]?(\d)[-\s]?(\d)[-\s]?(\d)', str(result))
            if not m: continue
            base4 = ''.join(m.groups())
            state = r.get(state_col,'') if state_col else ''
            game = r.get(game_col,'') if game_col else ''
            stream = r.get(stream_col,'') if stream_col else display_stream(state, game)
            date = r.get(date_col, '') if date_col else ''
            try: date = normalize_date(date)
            except Exception: date = ''
            core = eng.core_from_result(base4); member = eng.boxed_member(base4)
            rows.append({'play_date':date,'state':state,'game':game,'stream':stream,'result':base4,'member':member,'core':core,'is_watched8_core':core in WATCHED8})
        return pd.DataFrame(rows)

    rows=[]
    for line in text.splitlines():
        line=line.strip()
        if not line: continue
        parts=line.split('\t')
        if len(parts) >= 4:
            date_raw,state,game,result = parts[0],parts[1],parts[2],parts[3]
        else:
            # Loose fallback: date, state, game words, first 4-digit/dashed result.
            mres = re.search(r'(\d)\s*[- ]\s*(\d)\s*[- ]\s*(\d)\s*[- ]\s*(\d)', line)
            if not mres: continue
            result = mres.group(0)
            date_raw=''; state=''; game=''
        m = re.search(r'(\d)\s*[- ]\s*(\d)\s*[- ]\s*(\d)\s*[- ]\s*(\d)', str(result))
        if not m: continue
        base4=''.join(m.groups())
        try: date=normalize_date(date_raw)
        except Exception: date=''
        stream=display_stream(state, game)
        core=eng.core_from_result(base4); member=eng.boxed_member(base4)
        rows.append({'play_date':date,'state':state,'game':game,'stream':stream,'result':base4,'member':member,'core':core,'is_watched8_core':core in WATCHED8})
    return pd.DataFrame(rows)


def winners_from_history(hist: pd.DataFrame, play_date: str, watched_only=True) -> pd.DataFrame:
    date = normalize_date(play_date)
    d = hist[hist['draw_date'].eq(date)].copy()
    if d.empty:
        return pd.DataFrame(columns=['play_date','state','game','stream','result','member','core','is_watched8_core'])
    out = pd.DataFrame({
        'play_date': d['draw_date'].astype(str),
        'state': d.get('state',''),
        'game': d.get('game',''),
        'stream': d['stream'].astype(str),
        'result': d['base4'].astype(str).str.zfill(4),
        'member': d['member'].astype(str).str.zfill(4),
        'core': d['core'].astype(str).str.zfill(3),
    })
    out['is_watched8_core'] = out['core'].isin(WATCHED8)
    if watched_only:
        out = out[out['is_watched8_core']].copy()
    return out.reset_index(drop=True)


def choose_basis(step2: pd.DataFrame, basis: str) -> pd.DataFrame:
    d = step2.copy()
    num = lambda c: pd.to_numeric(d.get(c, 0), errors='coerce').fillna(0)
    if basis == 'all_step2_rows_cartesian_reference':
        return d
    if basis == 'final_x15_positive':
        return d[num('final_plus_transition_x15').gt(0)].copy()
    if basis == 'major_ge3':
        return d[num('major_support_count').ge(3)].copy()
    if basis == 'major_ge4':
        return d[num('major_support_count').ge(4)].copy()
    if basis == 'good_transition_ge1':
        return d[num('good_transition_count').ge(1)].copy()
    if basis == 'good_ge1_no_bad':
        return d[num('good_transition_count').ge(1) & num('bad_transition_count').le(0)].copy()
    if basis == 'major_ge3_and_good_ge1':
        return d[num('major_support_count').ge(3) & num('good_transition_count').ge(1)].copy()
    if basis == 'major_ge4_and_good_ge1':
        return d[num('major_support_count').ge(4) & num('good_transition_count').ge(1)].copy()
    if basis == 'profile_score_positive':
        return d[num('profile_final_member_score').gt(0)].copy()
    raise ValueError(f'Unknown bucket basis: {basis}')


RANK_STRATEGIES = [
    # Default V5 rank-lift set is deliberately small enough for 7-day runs.
    # The other formulas remain implemented below for later expansion if needed.
    'RANK_A_current_x15',
    'RANK_C_good_bad_x15',
    'RANK_D_support_good_bad_x15',
    'RANK_E_bucket_weighted_x15',
    'RANK_G_major_all_good_transition_x15',
]

def _ensure_rank_cols(out: pd.DataFrame) -> pd.DataFrame:
    defaults = {
        'final_plus_transition_x15':0, 'profile_final_member_score':0, 'transition_compat_score':0,
        'major_support_count':0, 'all_support_count':0, 'good_transition_count':0, 'bad_transition_count':0,
        'stream_gate_rank':9999, 'core_is_max':False, 'stream_at_or_above_median':False,
        'stream_above_median':False, 'stream_at_median':False, 'stream_above_mean':False,
        'core_at_or_above_median':False, 'core_above_median':False, 'core_at_median':False,
        'stream':'', 'core':'', 'member':''
    }
    for c,v in defaults.items():
        if c not in out.columns: out[c] = v
    # derive convenience booleans if not already present
    out['stream_at_or_above_median'] = out.get('stream_above_median', False).fillna(False).astype(bool) | out.get('stream_at_median', False).fillna(False).astype(bool)
    out['core_at_or_above_median'] = out.get('core_above_median', False).fillna(False).astype(bool) | out.get('core_at_median', False).fillna(False).astype(bool)
    return out

def _rank_sort_spec(strategy: str):
    # returns (columns, ascending) for stable best-to-worst sorting
    if strategy == 'RANK_A_current_x15':
        return ['final_plus_transition_x15','profile_final_member_score','major_support_count','all_support_count','stream_gate_rank','stream','core','member'], [False,False,False,False,True,True,True,True]
    if strategy == 'RANK_B_transition_score_first':
        return ['transition_compat_score','final_plus_transition_x15','major_support_count','good_transition_count','bad_transition_count','stream_gate_rank','stream','core','member'], [False,False,False,False,True,True,True,True,True]
    if strategy == 'RANK_C_good_bad_x15':
        return ['good_transition_count','bad_transition_count','final_plus_transition_x15','transition_compat_score','major_support_count','stream_gate_rank','stream','core','member'], [False,True,False,False,False,True,True,True,True]
    if strategy == 'RANK_D_support_good_bad_x15':
        return ['major_support_count','all_support_count','good_transition_count','bad_transition_count','final_plus_transition_x15','transition_compat_score','stream_gate_rank','stream','core','member'], [False,False,False,True,False,False,True,True,True,True]
    if strategy == 'RANK_E_bucket_weighted_x15':
        return ['rank_lift_bucket_score','final_plus_transition_x15','transition_compat_score','major_support_count','good_transition_count','bad_transition_count','stream_gate_rank','stream','core','member'], [False,False,False,False,False,True,True,True,True,True]
    if strategy == 'RANK_F_positive_transition_coremax_x15':
        return ['rank_lift_positive_transition','core_is_max','final_plus_transition_x15','transition_compat_score','major_support_count','good_transition_count','bad_transition_count','stream_gate_rank','stream','core','member'], [False,False,False,False,False,False,True,True,True,True,True]
    if strategy == 'RANK_G_major_all_good_transition_x15':
        return ['rank_lift_support_total','good_transition_count','transition_compat_score','final_plus_transition_x15','bad_transition_count','stream_gate_rank','stream','core','member'], [False,False,False,False,True,True,True,True,True]
    if strategy == 'RANK_H_low_bad_high_support_x15':
        return ['bad_transition_count','major_support_count','good_transition_count','final_plus_transition_x15','transition_compat_score','stream_gate_rank','stream','core','member'], [True,False,False,False,False,True,True,True,True]
    raise ValueError(f'Unknown rank strategy: {strategy}')

def add_rank_columns(d: pd.DataFrame, rank_strategy: str='RANK_A_current_x15', rank_col: str='rank_unified_best_to_worst') -> pd.DataFrame:
    out = _ensure_rank_cols(d.copy())
    num=lambda c: pd.to_numeric(out.get(c,0), errors='coerce').fillna(0)
    # fixed rank-lift fields from existing Step 2 columns only; no new mining
    out['rank_lift_positive_transition'] = num('transition_compat_score').gt(0).astype(int)
    out['rank_lift_support_total'] = num('major_support_count') + num('all_support_count')
    out['rank_lift_bucket_score'] = (
        out.get('core_is_max', False).fillna(False).astype(bool).astype(int)*3
        + out.get('core_at_or_above_median', False).fillna(False).astype(bool).astype(int)*1
        + out.get('stream_at_or_above_median', False).fillna(False).astype(bool).astype(int)*2
        + num('good_transition_count')
        - num('bad_transition_count')
    )
    cols, asc = _rank_sort_spec(rank_strategy)
    for c in cols:
        if c not in out.columns: out[c] = 0 if c not in ['stream','core','member'] else ''
    out = out.sort_values(cols, ascending=asc).copy()
    out[rank_col] = range(1, len(out)+1)
    out['rank_strategy'] = rank_strategy
    return out


def build_day_step2(hist: pd.DataFrame, prof: dict, play_date: str, history_through: str, step2_scope='watched8_all_members', gate_top=50, bucket_basis='final_x15_positive'):
    events = eng.derive_seed_events(hist, history_through, play_date)
    full = eng.build_full_enumeration(hist, events, prof)
    gate = eng.build_stream_gate(full, hist=hist, history_through=history_through, mode='baseline_n47')
    step2base = eng.build_step2_candidate_base(full, gate, use_top=int(gate_top), step2_scope=step2_scope)
    step2 = eng.apply_step2_transition(step2base, hist, history_through, x=15)
    basis_df = choose_basis(step2, bucket_basis)
    bucketed, core_counts, stream_counts = eng.add_rowcount_buckets(step2, basis_df=basis_df)
    bucketed = add_rank_columns(bucketed)
    return {'events': events, 'full': full, 'gate': gate, 'step2base': step2base, 'step2': bucketed, 'basis': basis_df, 'core_counts': core_counts, 'stream_counts': stream_counts}


def match_winners_to_step2(step2: pd.DataFrame, winners: pd.DataFrame) -> pd.DataFrame:
    if winners is None or winners.empty:
        return pd.DataFrame()
    d = step2.copy()
    for c in ['play_date','stream','core','member']:
        if c not in d.columns: d[c]=''
    d['play_date'] = d['play_date'].astype(str)
    d['stream_norm'] = d['stream'].astype(str).str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
    d['core'] = d['core'].astype(str).str.zfill(3)
    d['member'] = d['member'].astype(str).str.zfill(4)
    rows=[]
    for _,w in winners.iterrows():
        stream_norm = str(w.get('stream','')).lower().strip()
        stream_norm = re.sub(r'\s+', ' ', stream_norm)
        core = str(w.get('core','')).zfill(3)
        member = str(w.get('member','')).zfill(4)
        play_date = str(w.get('play_date',''))
        m = d[(d['play_date'].astype(str).eq(play_date)) & (d['stream_norm'].eq(stream_norm)) & (d['core'].eq(core)) & (d['member'].eq(member))].copy()
        if m.empty:
            # fallback by stream+member only if core mismatch due parsing should not happen
            m = d[(d['play_date'].astype(str).eq(play_date)) & (d['stream_norm'].eq(stream_norm)) & (d['member'].eq(member))].copy()
        if m.empty:
            rows.append({
                'play_date': play_date, 'winner_stream': w.get('stream',''), 'winner_result': w.get('result',''),
                'winner_member': member, 'winner_core': core, 'in_step2': False, 'rank_unified_best_to_worst': np.nan,
                'core_mean_bucket':'MISSING','core_median_bucket':'MISSING','stream_mean_bucket':'MISSING','stream_median_bucket':'MISSING'
            })
        else:
            r = m.iloc[0].to_dict()
            def bucket(prefix, system):
                if bool(r.get(f'{prefix}_above_{system}', False)): return f'above_{system}'
                if bool(r.get(f'{prefix}_at_{system}', False)): return f'at_{system}'
                if bool(r.get(f'{prefix}_below_{system}', False)): return f'below_{system}'
                return 'unknown'
            rows.append({
                'play_date': play_date, 'winner_stream': w.get('stream',''), 'winner_result': w.get('result',''),
                'winner_member': member, 'winner_core': core, 'in_step2': True,
                'rank_unified_best_to_worst': r.get('rank_unified_best_to_worst'),
                'step2_x15_rank': r.get('step2_x15_rank'),
                'seed': r.get('seed'), 'score_final_x15': r.get('final_plus_transition_x15'),
                'profile_final_member_score': r.get('profile_final_member_score'), 'transition_compat_score': r.get('transition_compat_score'),
                'major_support_count': r.get('major_support_count'), 'all_support_count': r.get('all_support_count'),
                'good_transition_count': r.get('good_transition_count'), 'bad_transition_count': r.get('bad_transition_count'),
                'core_row_count': r.get('core_row_count'), 'mean_core_row_count': r.get('mean_core_row_count'), 'median_core_row_count': r.get('median_core_row_count'),
                'stream_row_count': r.get('stream_row_count'), 'mean_stream_row_count': r.get('mean_stream_row_count'), 'median_stream_row_count': r.get('median_stream_row_count'),
                'core_mean_bucket': bucket('core','mean'), 'core_median_bucket': bucket('core','median'),
                'stream_mean_bucket': bucket('stream','mean'), 'stream_median_bucket': bucket('stream','median'),
                'core_is_max': r.get('core_is_max'), 'core_is_min': r.get('core_is_min'),
                'stream_is_max': r.get('stream_is_max'), 'stream_is_min': r.get('stream_is_min'),
            })
    return pd.DataFrame(rows)


def filter_mask(df: pd.DataFrame, qual='all_rows', core_filter='CORE_all', stream_filter='STREAM_all', transition_filter='TRANS_all'):
    d=df
    num=lambda c: pd.to_numeric(d.get(c,0), errors='coerce').fillna(0)
    mask=pd.Series(True, index=d.index)
    if qual == 'safe_no_bad': mask &= num('bad_transition_count').le(0)
    elif qual == 'aggressive_good1_no_bad': mask &= num('good_transition_count').ge(1) & num('bad_transition_count').le(0)
    elif qual == 'major_ge4': mask &= num('major_support_count').ge(4)
    elif qual == 'major_ge4_and_good_ge1': mask &= num('major_support_count').ge(4) & num('good_transition_count').ge(1)
    elif qual == 'major_ge4_and_aggressive': mask &= num('major_support_count').ge(4) & num('good_transition_count').ge(1) & num('bad_transition_count').le(0)
    elif qual == 'major_ge3_and_good_ge1': mask &= num('major_support_count').ge(3) & num('good_transition_count').ge(1)
    elif qual != 'all_rows': raise ValueError(f'Unknown qualification: {qual}')

    def core_or_stream_mask(prefix, f):
        if f.endswith('_all'):
            return pd.Series(True, index=d.index)
        pf = prefix.lower()
        if '_OR_' in f:
            left,right = f.split('_OR_',1)
            return core_or_stream_mask(prefix, left) | core_or_stream_mask(prefix, prefix+'_'+right)
        s=f.replace(prefix+'_','').lower()
        if s == 'is_max': col = pf+'_is_max'
        elif s == 'is_min': col = pf+'_is_min'
        else: col = pf+'_'+s
        if col in d.columns:
            return d[col].fillna(False).astype(bool)
        return pd.Series(False, index=d.index)

    mask &= core_or_stream_mask('CORE', core_filter)
    mask &= core_or_stream_mask('STREAM', stream_filter)

    tf = transition_filter
    if tf == 'TRANS_all': pass
    elif tf == 'TRANS_score_ge_0': mask &= num('transition_compat_score').ge(0)
    elif tf == 'TRANS_score_ge_0_25': mask &= num('transition_compat_score').ge(0.25)
    elif tf == 'TRANS_score_ge_0_5': mask &= num('transition_compat_score').ge(0.5)
    elif tf == 'TRANS_score_ge_1': mask &= num('transition_compat_score').ge(1)
    elif tf == 'GOOD_ge1': mask &= num('good_transition_count').ge(1)
    elif tf == 'GOOD_ge2': mask &= num('good_transition_count').ge(2)
    elif tf == 'GOOD_ge3': mask &= num('good_transition_count').ge(3)
    elif tf == 'GOOD_ge4': mask &= num('good_transition_count').ge(4)
    elif tf == 'BAD_eq0': mask &= num('bad_transition_count').le(0)
    elif tf == 'BAD_le1': mask &= num('bad_transition_count').le(1)
    elif tf == 'BAD_le2': mask &= num('bad_transition_count').le(2)
    elif tf == 'GOOD_ge1_BAD_le1': mask &= num('good_transition_count').ge(1) & num('bad_transition_count').le(1)
    elif tf == 'GOOD_ge1_BAD_le2': mask &= num('good_transition_count').ge(1) & num('bad_transition_count').le(2)
    elif tf == 'GOOD_ge2_BAD_le2': mask &= num('good_transition_count').ge(2) & num('bad_transition_count').le(2)
    elif tf == 'X15_ge_0': mask &= num('final_plus_transition_x15').ge(0)
    elif tf == 'X15_ge_10': mask &= num('final_plus_transition_x15').ge(10)
    elif tf == 'X15_ge_15': mask &= num('final_plus_transition_x15').ge(15)
    elif tf == 'X15_ge_20': mask &= num('final_plus_transition_x15').ge(20)
    else: raise ValueError(f'Unknown transition filter: {transition_filter}')
    return mask

def whatif_matrix(step2: pd.DataFrame, winners_loc: pd.DataFrame, quals=None, core_filters=None, stream_filters=None, transition_filters=None, rank_strategies=None, topn_list=(20,30,40,47,50,75,100), play_cap=50, lockdown_mode=True) -> pd.DataFrame:
    """Fast Step 3 what-if matrix with transition/score filters, rank-strategy lift tests, and Top-N quality."""
    quals = quals or (LOCKDOWN_QUAL_FILTERS if lockdown_mode else QUAL_FILTERS)
    core_filters = core_filters or (LOCKDOWN_CORE_FILTERS if lockdown_mode else CORE_FILTERS)
    stream_filters = stream_filters or (LOCKDOWN_STREAM_FILTERS if lockdown_mode else STREAM_FILTERS)
    transition_filters = transition_filters or (LOCKDOWN_TRANSITION_FILTERS if lockdown_mode else TRANSITION_FILTERS)
    rank_strategies = rank_strategies or RANK_STRATEGIES
    topn_list = tuple(sorted(set(int(x) for x in topn_list if int(x) > 0)))
    rows=[]
    winner_keys=set()
    if winners_loc is not None and not winners_loc.empty:
        for _,w in winners_loc[winners_loc.get('in_step2', False)==True].iterrows():
            winner_keys.add((str(w['play_date']), str(w['winner_stream']).lower().strip(), str(w['winner_core']).zfill(3), str(w['winner_member']).zfill(4)))
    d0=step2.copy()
    if d0.empty:
        return pd.DataFrame()
    d0['stream_norm']=d0['stream'].astype(str).str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
    d0['core']=d0['core'].astype(str).str.zfill(3)
    d0['member']=d0['member'].astype(str).str.zfill(4)
    d0['play_date']=d0['play_date'].astype(str)
    d0['_key'] = list(zip(d0['play_date'].astype(str), d0['stream_norm'].astype(str), d0['core'].astype(str), d0['member'].astype(str)))

    # Precompute masks on unsorted row index. Sorting happens separately per rank strategy.
    qual_masks={q: filter_mask(d0, q, 'CORE_all', 'STREAM_all', 'TRANS_all') for q in quals}
    core_masks={cf: filter_mask(d0, 'all_rows', cf, 'STREAM_all', 'TRANS_all') for cf in core_filters}
    stream_masks={sf: filter_mask(d0, 'all_rows', 'CORE_all', sf, 'TRANS_all') for sf in stream_filters}
    trans_masks={tf: filter_mask(d0, 'all_rows', 'CORE_all', 'STREAM_all', tf) for tf in transition_filters}

    # Fast path for dates where no watched-core winner reached Step 2.
    # Row counts still matter for affordability, but rank strategy cannot rescue a missing winner.
    if not winner_keys:
        base_rows=[]
        for q in quals:
            qm=qual_masks[q]
            for cf in core_filters:
                qcm=qm & core_masks[cf]
                if not qcm.any():
                    continue
                for sf in stream_filters:
                    qcms=qcm & stream_masks[sf]
                    if not qcms.any():
                        continue
                    for tf in transition_filters:
                        mask=qcms & trans_masks[tf]
                        rk=int(mask.sum())
                        if rk <= 0:
                            continue
                        for rank_strategy in rank_strategies:
                            row={'rank_strategy':rank_strategy,'qualification':q,'core_filter':cf,'stream_filter':sf,'transition_filter':tf,
                                 'rows_kept':rk,'under_or_equal_play_cap':rk <= int(play_cap),
                                 'winner_targets_in_step2':0,'winner_kept':0,'winner_missed_after_filter':0,
                                 'best_winner_rank':np.nan,'worst_winner_rank':np.nan,'avg_winner_rank':np.nan}
                            for topn in topn_list:
                                row[f'winner_top{topn}']=0
                            base_rows.append(row)
        return pd.DataFrame(base_rows)

    for rank_strategy in rank_strategies:
        d=add_rank_columns(d0, rank_strategy=rank_strategy)
        keys_array=d['_key'].tolist()
        # Masks need to align to the sorted index; reindex boolean masks to sorted index.
        qmasks={k:v.reindex(d.index).fillna(False).astype(bool) for k,v in qual_masks.items()}
        cmasks={k:v.reindex(d.index).fillna(False).astype(bool) for k,v in core_masks.items()}
        smasks={k:v.reindex(d.index).fillna(False).astype(bool) for k,v in stream_masks.items()}
        tmasks={k:v.reindex(d.index).fillna(False).astype(bool) for k,v in trans_masks.items()}
        for q in quals:
            qm=qmasks[q]
            for cf in core_filters:
                qcm = qm & cmasks[cf]
                if not qcm.any():
                    continue
                for sf in stream_filters:
                    qcms = qcm & smasks[sf]
                    if not qcms.any():
                        continue
                    for tf in transition_filters:
                        mask = qcms & tmasks[tf]
                        idx=np.flatnonzero(mask.to_numpy())
                        keep_keys=[keys_array[i] for i in idx] if len(idx) else []
                        kept_set=set(keep_keys)
                        found = sorted(winner_keys & kept_set)
                        ranks=[]
                        if found:
                            rank_lookup={k: r+1 for r,k in enumerate(keep_keys)}
                            ranks=[rank_lookup[k] for k in found if k in rank_lookup]
                        row={
                            'rank_strategy':rank_strategy,
                            'qualification':q,'core_filter':cf,'stream_filter':sf,'transition_filter':tf,
                            'rows_kept':int(len(idx)),
                            'under_or_equal_play_cap': int(len(idx)) <= int(play_cap),
                            'winner_targets_in_step2':len(winner_keys),
                            'winner_kept':len(found),
                            'winner_missed_after_filter': max(len(winner_keys)-len(found),0),
                            'best_winner_rank': min(ranks) if ranks else np.nan,
                            'worst_winner_rank': max(ranks) if ranks else np.nan,
                            'avg_winner_rank': float(np.mean(ranks)) if ranks else np.nan,
                        }
                        for topn in topn_list:
                            row[f'winner_top{topn}']=sum(1 for r in ranks if r <= int(topn))
                        rows.append(row)
    out=pd.DataFrame(rows)
    if not out.empty:
        out['capture_rate_kept'] = np.where(out['winner_targets_in_step2'].gt(0), out['winner_kept']/out['winner_targets_in_step2'], np.nan)
        for topn in topn_list:
            c=f'winner_top{topn}'
            out[f'capture_rate_top{topn}'] = np.where(out['winner_targets_in_step2'].gt(0), out[c]/out['winner_targets_in_step2'], np.nan)
        sort_cols=[]; ascending=[]
        if 'winner_top50' in out.columns:
            sort_cols.append('winner_top50'); ascending.append(False)
        sort_cols += ['under_or_equal_play_cap','winner_kept','rows_kept','avg_winner_rank']
        ascending += [False,False,True,True]
        out=out.sort_values(sort_cols, ascending=ascending)
    return out

def summarize_topn_quality(what_all: pd.DataFrame, play_cap=50) -> pd.DataFrame:
    if what_all is None or what_all.empty:
        return pd.DataFrame()
    top_cols=[c for c in what_all.columns if c.startswith('winner_top')]
    agg={
        'dates_tested':('play_date','nunique'),
        'total_rows_kept':('rows_kept','sum'),
        'avg_rows_kept':('rows_kept','mean'),
        'max_rows_kept':('rows_kept','max'),
        'winner_targets':('winner_targets_in_step2','sum'),
        'winners_kept':('winner_kept','sum'),
        'avg_winner_rank_mean':('avg_winner_rank','mean'),
        'worst_winner_rank_max':('worst_winner_rank','max'),
    }
    for c in top_cols:
        agg[c]=(c,'sum')
    combo=what_all.groupby(['rank_strategy','qualification','core_filter','stream_filter','transition_filter'], as_index=False).agg(**agg)
    combo['avg_rows_under_or_equal_play_cap'] = combo['avg_rows_kept'].le(int(play_cap))
    combo['max_rows_under_or_equal_play_cap'] = combo['max_rows_kept'].le(int(play_cap))
    combo['capture_rate_kept']=np.where(combo['winner_targets'].gt(0), combo['winners_kept']/combo['winner_targets'], np.nan)
    if 'winner_top50' in what_all.columns:
        dayhits = what_all.assign(day_has_top50=what_all['winner_top50'].gt(0).astype(int)).groupby(['rank_strategy','qualification','core_filter','stream_filter','transition_filter'])['day_has_top50'].sum().reset_index(name='days_with_top50_winner')
        combo = combo.merge(dayhits, on=['rank_strategy','qualification','core_filter','stream_filter','transition_filter'], how='left')
    else:
        combo['days_with_top50_winner'] = 0
    combo['days_with_top50_winner'] = combo['days_with_top50_winner'].fillna(0).astype(int)
    combo['weekly_75pct_days_top50'] = np.where(combo['dates_tested'].gt(0), combo['days_with_top50_winner'] / combo['dates_tested'], np.nan)
    for c in top_cols:
        n=c.replace('winner_top','')
        combo[f'capture_rate_top{n}']=np.where(combo['winner_targets'].gt(0), combo[c]/combo['winner_targets'], np.nan)
    sort_cols=[]; ascending=[]
    if 'days_with_top50_winner' in combo.columns:
        sort_cols += ['days_with_top50_winner']; ascending += [False]
    if 'winner_top50' in combo.columns:
        sort_cols += ['winner_top50']; ascending += [False]
    sort_cols += ['winners_kept','avg_rows_under_or_equal_play_cap','avg_rows_kept','avg_winner_rank_mean']
    ascending += [False,False,True,True]
    return combo.sort_values(sort_cols, ascending=ascending)

def summarize_rank_lift(what_all: pd.DataFrame, play_cap=50) -> pd.DataFrame:
    """Roll up how each ranking formula performs independent of the exact Step 3 combo."""
    if what_all is None or what_all.empty or 'rank_strategy' not in what_all.columns:
        return pd.DataFrame()
    # Best per date per rank strategy under strict play cap, then summarize.
    d=what_all.copy()
    if 'winner_top50' not in d.columns:
        d['winner_top50']=0
    strict=d[d['rows_kept'].le(int(play_cap))].copy()
    if strict.empty:
        strict=d.copy()
    strict= strict.sort_values(['rank_strategy','play_date','winner_top50','winner_kept','rows_kept','avg_winner_rank'], ascending=[True,True,False,False,True,True])
    best_day = strict.groupby(['rank_strategy','play_date'], as_index=False).head(1)
    agg=best_day.groupby('rank_strategy', as_index=False).agg(
        dates_tested=('play_date','nunique'),
        days_with_top50_winner=('winner_top50', lambda x: int((pd.to_numeric(x, errors='coerce').fillna(0)>0).sum())),
        total_top50_winners=('winner_top50','sum'),
        winners_kept=('winner_kept','sum'),
        avg_rows_kept=('rows_kept','mean'),
        max_rows_kept=('rows_kept','max'),
        avg_winner_rank_mean=('avg_winner_rank','mean'),
        worst_winner_rank_max=('worst_winner_rank','max'),
    )
    agg['weekly_75pct_days_top50']=np.where(agg['dates_tested'].gt(0), agg['days_with_top50_winner']/agg['dates_tested'], np.nan)
    return agg.sort_values(['days_with_top50_winner','total_top50_winners','avg_rows_kept','avg_winner_rank_mean'], ascending=[False,False,True,True])

def replay_audit(history_path, profile_dir, out_dir, start_date, end_date, winners_path=None, use_history_winners=True, exclude_az_md=True, step2_scope='watched8_all_members', bucket_basis='final_x15_positive', gate_top=50, max_dates=31, progress_cb=None, audit_watched_only=True, play_cap=50, write_support_debug=False):
    out=Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    hist0=eng.read_history(history_path)
    hist, exclusion_audit=eng.apply_exclusions(hist0, exclude_az_md=exclude_az_md)
    prof=eng.load_profiles(profile_dir)
    start=pd.to_datetime(start_date); end=pd.to_datetime(end_date)
    dates=[d.strftime('%Y-%m-%d') for d in pd.date_range(start,end,freq='D')]
    if len(dates) > int(max_dates):
        dates=dates[:int(max_dates)]
    external_winners = read_winners_any(winners_path) if winners_path else pd.DataFrame()
    if audit_watched_only and not external_winners.empty and 'is_watched8_core' in external_winners.columns:
        external_winners = external_winners[external_winners['is_watched8_core'].astype(bool)].copy()
    all_loc=[]; all_summ=[]; all_what=[]; logs=[]
    for i,play_date in enumerate(dates, start=1):
        t0=time.time()
        through=(pd.to_datetime(play_date)-pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        if progress_cb: progress_cb(i, len(dates), play_date, 'building Step 0-2')
        try:
            parts=build_day_step2(hist, prof, play_date, through, step2_scope=step2_scope, gate_top=gate_top, bucket_basis=bucket_basis)
            step2=parts['step2']
            # Optional support-count/join diagnostics. Disabled by default in V5 rank-lift runs for speed.
            if write_support_debug:
                try:
                    sd.write_support_debug_outputs(out, profile_dir, play_date, parts.get('full', pd.DataFrame()), parts.get('step2base', pd.DataFrame()), step2)
                except Exception as _support_debug_error:
                    logs.append({'play_date':play_date,'stage':'support_debug','error':str(_support_debug_error)})
            # Winners: use external if present for date, otherwise use history winners if requested.
            if not external_winners.empty:
                winners=external_winners[external_winners['play_date'].astype(str).eq(play_date)].copy()
            else:
                winners=pd.DataFrame()
            if winners.empty and use_history_winners:
                winners=winners_from_history(hist, play_date, watched_only=True)
            loc=match_winners_to_step2(step2, winners) if not winners.empty else pd.DataFrame()
            if not loc.empty: all_loc.append(loc)
            wm=whatif_matrix(step2, loc, play_cap=play_cap) if not loc.empty else pd.DataFrame()
            if not wm.empty:
                wm.insert(0,'play_date',play_date); all_what.append(wm)
            summ={
                'play_date': play_date, 'history_through': through, 'status':'OK', 'seconds':round(time.time()-t0,2),
                'history_rows_after_exclusion': len(hist), 'seed_streams': parts['events']['stream'].nunique(),
                'step1_rows': len(parts['full']), 'stream_gate_rows': int(parts['gate'][f'in_stream_gate_top{gate_top}'].sum()) if f'in_stream_gate_top{gate_top}' in parts['gate'].columns else len(parts['gate']),
                'step2_rows': len(step2), 'bucket_basis_rows': len(parts['basis']), 'actual_watched_core_winners': len(winners),
                'winners_found_in_step2': int(loc['in_step2'].sum()) if not loc.empty and 'in_step2' in loc.columns else 0,
            }
            all_summ.append(summ)
            # Save one per-date compact step2 for deeper inspection but not full enumeration.
            step2_cols=[c for c in ['play_date','stream','seed','core','member','rank_unified_best_to_worst','final_plus_transition_x15','profile_final_member_score','transition_compat_score','major_support_count','all_support_count','good_transition_count','bad_transition_count','core_row_count','stream_row_count','core_above_mean','core_at_mean','core_below_mean','core_above_median','core_at_median','core_below_median','stream_above_mean','stream_at_mean','stream_below_mean','stream_above_median','stream_at_median','stream_below_median','core_is_max','core_is_min','stream_is_max','stream_is_min'] if c in step2.columns]
            step2[step2_cols].to_csv(out/f'STEP2_BUCKETED_ROWS_{play_date}.csv', index=False)
        except Exception as e:
            all_summ.append({'play_date':play_date,'history_through':through,'status':'FAILED','error':repr(e),'seconds':round(time.time()-t0,2)})
            logs.append({'play_date':play_date,'error':repr(e)})
    summary=pd.DataFrame(all_summ)
    loc_all=pd.concat(all_loc, ignore_index=True) if all_loc else pd.DataFrame()
    what_all=pd.concat(all_what, ignore_index=True) if all_what else pd.DataFrame()
    exclusion_audit.to_csv(out/'00_STEP0_EXCLUSION_AUDIT.csv', index=False)
    summary.to_csv(out/'00_AUDIT_RUN_SUMMARY.csv', index=False)
    loc_all.to_csv(out/'01_WINNER_LOCATION_IN_STEP2_BUCKETS.csv', index=False)
    what_all.to_csv(out/'02_STEP3_FILTER_WHATIF_WINNER_SURVIVAL.csv', index=False)
    pd.DataFrame(logs).to_csv(out/'99_FAILURE_LOG.csv', index=False)
    # Rollup combo performance with rank quality and Top-N counts.
    if not what_all.empty:
        combo=summarize_topn_quality(what_all, play_cap=play_cap)
        combo.to_csv(out/'03_BEST_STEP3_COMBOS_BY_CAPTURE_AND_PLAYCOUNT.csv', index=False)
        rank_lift=summarize_rank_lift(what_all, play_cap=play_cap)
        rank_lift.to_csv(out/'08_RANK_LIFT_STRATEGY_SUMMARY.csv', index=False)
        # Separate affordable subset for the actual play-cap goal.
        affordable=combo[combo['avg_rows_kept'].le(int(play_cap))].copy()
        affordable.to_csv(out/f'04_AFFORDABLE_COMBOS_AVG_ROWS_LE_{int(play_cap)}.csv', index=False)
        strict=combo[combo['max_rows_kept'].le(int(play_cap))].copy()
        strict.to_csv(out/f'05_STRICT_COMBOS_MAX_ROWS_LE_{int(play_cap)}.csv', index=False)
        target_days = int(np.ceil(0.75 * max(combo['dates_tested'].max(), 1))) if not combo.empty else 0
        lockdown = combo[(combo['max_rows_kept'].le(int(play_cap))) & (combo['days_with_top50_winner'].ge(target_days))].copy() if 'days_with_top50_winner' in combo.columns else pd.DataFrame()
        lockdown.to_csv(out/f'06_LOCKDOWN_CANDIDATES_75PCT_DAYS_TOP50_MAX_ROWS_LE_{int(play_cap)}.csv', index=False)
        near = combo[(combo['avg_rows_kept'].le(int(play_cap)*1.5)) & (combo['days_with_top50_winner'].ge(max(target_days-1,1)))].copy() if 'days_with_top50_winner' in combo.columns else pd.DataFrame()
        near.to_csv(out/f'07_NEAR_MISS_CANDIDATES_AVG_ROWS_LE_{int(play_cap)}x1_5.csv', index=False)
    # zip
    zip_path=out.parent/(f'{BUILD_ID}_{dates[0]}_TO_{dates[-1]}_OUTPUTS.zip' if dates else f'{BUILD_ID}_OUTPUTS.zip')
    with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file(): z.write(p, p.relative_to(out))
    return {'summary':summary,'winner_locations':loc_all,'whatif':what_all,'zip_path':str(zip_path),'out_dir':str(out)}

# -----------------------------------------------------------------------------
# EMERGENCY DAILY PLAYER V1 — V5 lane defaults
# -----------------------------------------------------------------------------
EMERGENCY_BUILD_ID = 'EMERGENCY_DAILY_PLAYER_V1_5_DECISION_LAYER_AUDIT'

EMERGENCY_LANES = [
    {
        'lane_id': 'L1_MAJOR4_STREAMMAX_COREMED',
        'label': 'Major4 + stream max + core median',
        'rank_strategy': 'RANK_A_current_x15',
        'qualification': 'major_ge4_and_good_ge1',
        'core_filter': 'CORE_at_or_above_median',
        'stream_filter': 'STREAM_is_max',
        'transition_filter': 'TRANS_all',
        'why': 'Caught 06/12 in 3 rows during V5 replay.'
    },
    {
        'lane_id': 'L2_MAJOR3_COREMAX_STREAMMED_GOODBAD',
        'label': 'Major3 + core max + stream median + good/bad',
        'rank_strategy': 'RANK_A_current_x15',
        'qualification': 'major_ge3_and_good_ge1',
        'core_filter': 'CORE_is_max',
        'stream_filter': 'STREAM_at_or_above_median',
        'transition_filter': 'GOOD_ge1_BAD_le2',
        'why': 'Caught 06/13 in 15 rows during V5 replay.'
    },
    {
        'lane_id': 'L3_MAJOR4_COREMED_LOWSTREAM_TRANS1',
        'label': 'Major4 + core median + low stream + transition ≥ 1',
        'rank_strategy': 'RANK_A_current_x15',
        'qualification': 'major_ge4_and_good_ge1',
        'core_filter': 'CORE_at_or_above_median',
        'stream_filter': 'STREAM_at_or_below_mean',
        'transition_filter': 'TRANS_score_ge_1',
        'why': 'Caught 06/16 in 13 rows during V5 replay.'
    },
    {
        'lane_id': 'L4_MAJOR3_COREMED_STREAMMAX_GOODBAD',
        'label': 'Major3 + core median + stream max + good/bad',
        'rank_strategy': 'RANK_A_current_x15',
        'qualification': 'major_ge3_and_good_ge1',
        'core_filter': 'CORE_at_or_above_median',
        'stream_filter': 'STREAM_is_max',
        'transition_filter': 'GOOD_ge1_BAD_le2',
        'why': 'Caught 06/17 in 25 rows during V5 replay.'
    },
    {
        'lane_id': 'L5_MAJOR3_COREMAX_LOWSTREAM_X15',
        'label': 'Major3 + core max + low stream + X15 ≥ 15',
        'rank_strategy': 'RANK_E_bucket_weighted_x15',
        'qualification': 'major_ge3_and_good_ge1',
        'core_filter': 'CORE_is_max',
        'stream_filter': 'STREAM_at_or_below_mean',
        'transition_filter': 'X15_ge_15',
        'why': 'Caught 06/18 in 34 rows during V5 replay.'
    },
    {
        'lane_id': 'L6_OREGON_LOCK_STREAMMAX_TRANS1',
        'label': 'Oregon-style lockdown: Major3 + stream max + transition ≥ 1',
        'rank_strategy': 'RANK_C_good_bad_x15',
        'qualification': 'major_ge3_and_good_ge1',
        'core_filter': 'CORE_all',
        'stream_filter': 'STREAM_is_max',
        'transition_filter': 'TRANS_score_ge_1',
        'why': 'Caught the 06/19 Oregon validation winner at rank #4 in a 14-row V5 test.'
    },
]


def _as_bool_series(s, index):
    if s is None:
        return pd.Series(False, index=index)
    if getattr(s, 'dtype', None) == bool:
        return s.fillna(False)
    return s.astype(str).str.lower().isin(['true','1','yes','y'])


def _normalize_play_cols(d: pd.DataFrame) -> pd.DataFrame:
    out = d.copy()
    if 'core' in out.columns:
        out['core'] = out['core'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(3)
    if 'member' in out.columns:
        out['member'] = out['member'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(4)
    if 'seed' in out.columns:
        out['seed'] = out['seed'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(4)
    return out


def build_emergency_daily_playlist(
    history_path,
    profile_dir,
    out_dir,
    play_date,
    history_through=None,
    exclude_az_md=True,
    step2_scope='watched8_all_members',
    bucket_basis='final_x15_positive',
    gate_top=50,
    play_cap=50,
    lane_top_n=50,
    lanes=None,
    write_full_step2=True,
):
    """Build the V5 emergency daily playable box playlist.

    This is a production-style daily builder, not a winner auditor. It uses no future winners.
    It builds Step 0-2 blind from history through `history_through`, runs the fixed V5
    emergency lanes, merges/dedupes rows, and caps the final playlist.
    """
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    lanes = lanes or EMERGENCY_LANES
    play_date = pd.to_datetime(play_date).strftime('%Y-%m-%d')
    if history_through is None or str(history_through).strip() == '':
        history_through = (pd.to_datetime(play_date) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    else:
        history_through = pd.to_datetime(history_through).strftime('%Y-%m-%d')

    hist0 = eng.read_history(history_path)
    hist, exclusion_audit = eng.apply_exclusions(hist0, exclude_az_md=exclude_az_md)
    prof = eng.load_profiles(profile_dir)
    parts = build_day_step2(hist, prof, play_date, history_through, step2_scope=step2_scope, gate_top=int(gate_top), bucket_basis=bucket_basis)
    step2 = _normalize_play_cols(parts['step2'])

    lane_rows = []
    lane_summary = []
    for lane in lanes:
        ranked = add_rank_columns(step2, rank_strategy=lane['rank_strategy']).copy()
        ranked = _normalize_play_cols(ranked)
        mask = filter_mask(ranked, lane['qualification'], lane['core_filter'], lane['stream_filter'], lane['transition_filter'])
        selected = ranked[mask].copy()
        selected['lane_rank'] = range(1, len(selected) + 1)
        selected['lane_id'] = lane['lane_id']
        selected['lane_priority'] = int(lanes.index(lane) + 1)
        selected['lane_label'] = lane['label']
        selected['lane_rank_strategy'] = lane['rank_strategy']
        selected['lane_qualification'] = lane['qualification']
        selected['lane_core_filter'] = lane['core_filter']
        selected['lane_stream_filter'] = lane['stream_filter']
        selected['lane_transition_filter'] = lane['transition_filter']
        selected['lane_reason'] = lane.get('why', '')
        if int(lane_top_n) > 0:
            selected = selected.head(int(lane_top_n)).copy()
        lane_summary.append({
            'lane_id': lane['lane_id'],
            'lane_priority': int(lanes.index(lane) + 1),
            'lane_label': lane['label'],
            'rank_strategy': lane['rank_strategy'],
            'qualification': lane['qualification'],
            'core_filter': lane['core_filter'],
            'stream_filter': lane['stream_filter'],
            'transition_filter': lane['transition_filter'],
            'rows_selected_before_dedupe': len(selected),
            'reason_from_v5_replay': lane.get('why', ''),
        })
        if not selected.empty:
            lane_rows.append(selected)

    if lane_rows:
        lane_all = pd.concat(lane_rows, ignore_index=True)
    else:
        lane_all = pd.DataFrame()

    if lane_all.empty:
        final = pd.DataFrame()
    else:
        key_cols = ['stream','seed','core','member']
        keep_cols = [c for c in [
            'play_date','history_through','stream','seed','core','member','final_plus_transition_x15',
            'profile_final_member_score','transition_compat_score','major_support_count','all_support_count',
            'good_transition_count','bad_transition_count','core_row_count','stream_row_count',
            'core_is_max','stream_is_max','rank_unified_best_to_worst','rank_lift_bucket_score'
        ] if c in lane_all.columns]
        # Aggregate lane evidence.
        agg_rows=[]
        for key, g in lane_all.groupby(key_cols, dropna=False):
            g=g.copy()
            first=g.iloc[0]
            lane_ids=sorted(g['lane_id'].astype(str).unique())
            lane_labels=sorted(g['lane_label'].astype(str).unique())
            rank_strats=sorted(g['lane_rank_strategy'].astype(str).unique())
            row={c:first.get(c) for c in keep_cols if c not in key_cols}
            for col,val in zip(key_cols,key):
                row[col]=val
            row['lane_count']=len(lane_ids)
            row['lane_entries']=len(g)
            row['lane_ids']='; '.join(lane_ids)
            row['lane_labels']='; '.join(lane_labels)
            row['rank_strategies']='; '.join(rank_strats)
            lane_rank_num = pd.to_numeric(g['lane_rank'], errors='coerce')
            row['best_lane_rank']=int(lane_rank_num.min())
            row['avg_lane_rank']=float(lane_rank_num.mean())
            best_g = g[lane_rank_num == lane_rank_num.min()].copy()
            row['best_lane_ids']='; '.join(sorted(best_g['lane_id'].astype(str).unique()))
            row['min_lane_priority']=int(pd.to_numeric(g.get('lane_priority', 99), errors='coerce').min()) if 'lane_priority' in g.columns else 99
            row['best_global_rank']=int(pd.to_numeric(g.get('rank_unified_best_to_worst', 9999), errors='coerce').min()) if 'rank_unified_best_to_worst' in g.columns else 9999
            for c in ['final_plus_transition_x15','profile_final_member_score','transition_compat_score','major_support_count','all_support_count','good_transition_count','rank_lift_bucket_score']:
                if c in g.columns:
                    row[c]=pd.to_numeric(g[c], errors='coerce').max()
            if 'bad_transition_count' in g.columns:
                row['bad_transition_count']=pd.to_numeric(g['bad_transition_count'], errors='coerce').min()
            agg_rows.append(row)
        final = pd.DataFrame(agg_rows)
        num=lambda c: pd.to_numeric(final.get(c,0), errors='coerce').fillna(0)
        final['consensus_score'] = (
            num('lane_count') * 1000
            + (51 - num('best_lane_rank').clip(lower=1, upper=50)) * 20
            + num('rank_lift_bucket_score') * 25
            + num('major_support_count') * 12
            + num('all_support_count') * 5
            + num('good_transition_count') * 20
            - num('bad_transition_count') * 15
            + num('transition_compat_score') * 40
            + num('final_plus_transition_x15')
        )
        # V1.3: keep the per-lane lists as the comparison authority.
        # The combined list is only a secondary convenience list and is driven by
        # best rank in any single lane, so strong one-lane winners are not buried
        # by a consensus-count sort.
        final['legacy_consensus_rank_marker'] = final['consensus_score']
        consensus_legacy = final.sort_values([
            'consensus_score','lane_count','best_lane_rank','final_plus_transition_x15',
            'major_support_count','all_support_count','good_transition_count','bad_transition_count','stream','core','member'
        ], ascending=[False,False,True,False,False,False,False,True,True,True,True]).copy()
        consensus_legacy['LegacyConsensusRank'] = range(1, len(consensus_legacy)+1)

        final = final.sort_values([
            'best_lane_rank','min_lane_priority','lane_count','final_plus_transition_x15',
            'major_support_count','all_support_count','good_transition_count','bad_transition_count','stream','core','member'
        ], ascending=[True,True,False,False,False,False,False,True,True,True,True]).copy()
        final['FinalRank'] = range(1, len(final)+1)
        final = final.head(int(play_cap)).copy()
        final['PLAY_DATE'] = play_date
        final['HISTORY_THROUGH'] = history_through
        # Friendly order.
        first_cols = ['FinalRank','PLAY_DATE','HISTORY_THROUGH','stream','seed','core','member','best_lane_rank','best_lane_ids','lane_count','lane_ids','rank_strategies','consensus_score']
        final = final[[c for c in first_cols if c in final.columns] + [c for c in final.columns if c not in first_cols]]

    lane_summary_df = pd.DataFrame(lane_summary)
    run_summary = pd.DataFrame([{
        'build': EMERGENCY_BUILD_ID,
        'play_date': play_date,
        'history_through': history_through,
        'history_rows_after_exclusion': len(hist),
        'exclude_az_md': bool(exclude_az_md),
        'step2_scope': step2_scope,
        'bucket_basis': bucket_basis,
        'stream_gate_top': int(gate_top),
        'step2_rows': len(step2),
        'lane_rows_before_dedupe': len(lane_all),
        'unique_lane_rows_before_cap': len(lane_all.drop_duplicates(['stream','seed','core','member'])) if not lane_all.empty else 0,
        'merged_candidate_rows': len(final),
        'final_rows_deprecated_name': len(final),
        'play_cap': int(play_cap),
        'lane_top_n': int(lane_top_n),
        'profile_files_nonempty': sum(1 for _k,_v in prof.items() if hasattr(_v, 'empty') and not _v.empty) if isinstance(prof, dict) else np.nan,
        'profile_files_expected': len(prof) if isinstance(prof, dict) else np.nan,
        'support_major_nonzero_step2': int(pd.to_numeric(step2.get('major_support_count', 0), errors='coerce').fillna(0).gt(0).sum()) if len(step2) else 0,
        'support_all_nonzero_step2': int(pd.to_numeric(step2.get('all_support_count', 0), errors='coerce').fillna(0).gt(0).sum()) if len(step2) else 0,
    }])

    run_summary.to_csv(out/'00_RUN_SUMMARY.csv', index=False)
    if len(lane_all) == 0:
        (out/'00_ZERO_PLAYLIST_DIAGNOSIS.txt').write_text(
            'FINAL ROWS = 0. The usual cause is that the profile/support CSV files were not uploaded with the app, so major_support_count/good_transition_count stay at zero. Upload the profiles/ folder from the zip, or put the V6_8CORE_*.csv and related profile CSVs at the repo root. Check 00_RUN_SUMMARY.csv: profile_files_nonempty and support_major_nonzero_step2 must be greater than 0.\n',
            encoding='utf-8'
        )
    exclusion_audit.to_csv(out/'00_STEP0_EXCLUSION_AUDIT.csv', index=False)
    lane_summary_df.to_csv(out/'01_LANE_SUMMARY.csv', index=False)
    final.to_csv(out/'02_MERGED_LANE_CANDIDATES_BY_BEST_LANE_RANK.csv', index=False)
    final.to_csv(out/'02A_COMBINED_BY_BEST_LANE_RANK.csv', index=False)  # retained alias for comparison
    try:
        consensus_legacy.head(int(play_cap)).to_csv(out/'02B_COMBINED_BY_CONSENSUS_LEGACY_DO_NOT_USE_AS_PRIMARY.csv', index=False)
    except Exception:
        pd.DataFrame().to_csv(out/'02B_COMBINED_BY_CONSENSUS_LEGACY_DO_NOT_USE_AS_PRIMARY.csv', index=False)
    if not lane_all.empty:
        lane_all.to_csv(out/'03_ALL_LANE_ROWS_BEFORE_DEDUPE.csv', index=False)
        lane_dir = out / '05_SEPARATE_LANE_PLAYLISTS'
        lane_dir.mkdir(exist_ok=True)
        for lane in lanes:
            lid = lane['lane_id']
            one = lane_all[lane_all['lane_id'].astype(str) == str(lid)].copy()
            if not one.empty:
                one = one.head(int(play_cap)).copy()
                one.insert(0, 'LanePlayRank', range(1, len(one)+1))
                one.insert(1, 'PLAY_DATE', play_date)
                one.insert(2, 'HISTORY_THROUGH', history_through)
                safe_lid = ''.join(ch if ch.isalnum() or ch in ['_','-'] else '_' for ch in str(lid))
                one.to_csv(lane_dir / f'{safe_lid}.csv', index=False)
                # Simple printable lane file.
                lines_lane = [f"{lid} | {lane.get('label','')}", f"PLAY_DATE: {play_date}", f"HISTORY_THROUGH: {history_through}", "Rank | Stream | Seed | Core | Member | LaneRank | X15 | Major | Good | Bad"]
                for _, rr in one.iterrows():
                    lines_lane.append(f"{int(rr.get('LanePlayRank',0)):02d} | {rr.get('stream','')} | {str(rr.get('seed','')).zfill(4)} | {str(rr.get('core','')).zfill(3)} | {str(rr.get('member','')).zfill(4)} | {int(rr.get('lane_rank',0))} | {float(rr.get('final_plus_transition_x15',0)):.3f} | {int(rr.get('major_support_count',0))} | {int(rr.get('good_transition_count',0))} | {int(rr.get('bad_transition_count',0))}")
                (lane_dir / f'{safe_lid}.txt').write_text('\n'.join(lines_lane), encoding='utf-8')
    else:
        pd.DataFrame().to_csv(out/'03_ALL_LANE_ROWS_BEFORE_DEDUPE.csv', index=False)
    if write_full_step2:
        step2.to_csv(out/'04_FULL_STEP2_BUCKETED_ROWS.csv', index=False)


    # Stream-overlap audit: Top 30 from each lane, grouped by stream.
    # This is NOT a final member selector. It answers whether multiple lanes agree on a stream,
    # while preserving all lane-qualified core/member rows for that stream.
    stream_overlap_streams = pd.DataFrame()
    stream_overlap_rows = pd.DataFrame()
    if not lane_all.empty:
        top30 = lane_all[pd.to_numeric(lane_all.get('lane_rank', 9999), errors='coerce').fillna(9999).le(30)].copy()
        if not top30.empty:
            top30 = _normalize_play_cols(top30)
            grp_rows=[]
            for stream, g in top30.groupby('stream', dropna=False):
                g=g.copy()
                lane_ids=sorted(g['lane_id'].astype(str).unique()) if 'lane_id' in g.columns else []
                lane_labels=sorted(g['lane_label'].astype(str).unique()) if 'lane_label' in g.columns else []
                lane_rank_num=pd.to_numeric(g.get('lane_rank', 9999), errors='coerce').fillna(9999)
                core_members=sorted((g['core'].astype(str).str.zfill(3)+'/'+g['member'].astype(str).str.zfill(4)).unique()) if {'core','member'}.issubset(g.columns) else []
                seeds=sorted(g['seed'].astype(str).str.zfill(4).unique()) if 'seed' in g.columns else []
                grp_rows.append({
                    'stream': stream,
                    'appears_in_lanes_top30': len(lane_ids),
                    'lane_ids_top30': '; '.join(lane_ids),
                    'lane_labels_top30': '; '.join(lane_labels),
                    'rows_for_stream_top30': len(g),
                    'best_lane_rank_for_stream': int(lane_rank_num.min()),
                    'avg_lane_rank_for_stream': float(lane_rank_num.mean()),
                    'core_member_options_top30': '; '.join(core_members),
                    'unique_core_member_options': len(core_members),
                    'seeds_seen': '; '.join(seeds),
                    'max_final_plus_transition_x15': float(pd.to_numeric(g.get('final_plus_transition_x15',0), errors='coerce').fillna(0).max()),
                    'max_major_support_count': int(pd.to_numeric(g.get('major_support_count',0), errors='coerce').fillna(0).max()),
                    'max_good_transition_count': int(pd.to_numeric(g.get('good_transition_count',0), errors='coerce').fillna(0).max()),
                    'review_flag': 'STREAM_OVERLAP_2PLUS_LANES' if len(lane_ids) >= 2 else 'SINGLE_LANE_ONLY',
                })
            stream_overlap_streams = pd.DataFrame(grp_rows).sort_values([
                'appears_in_lanes_top30','best_lane_rank_for_stream','rows_for_stream_top30','max_final_plus_transition_x15','stream'
            ], ascending=[False, True, False, False, True]).reset_index(drop=True)
            stream_overlap_streams.insert(0, 'StreamOverlapRank', range(1, len(stream_overlap_streams)+1))
            top30['_stream_lane_count_top30'] = top30['stream'].map(stream_overlap_streams.set_index('stream')['appears_in_lanes_top30'].to_dict())
            top30['_stream_overlap_rank'] = top30['stream'].map(stream_overlap_streams.set_index('stream')['StreamOverlapRank'].to_dict())
            stream_overlap_rows = top30.sort_values(['_stream_overlap_rank','stream','lane_id','lane_rank','core','member'], ascending=[True, True, True, True, True, True]).copy()
    stream_overlap_streams.to_csv(out/'06A_STREAM_OVERLAP_TOP30_STREAMS.csv', index=False)
    stream_overlap_rows.to_csv(out/'06B_STREAM_OVERLAP_TOP30_ROWS.csv', index=False)
    # Printable stream-overlap summary.
    lines_ov=[]
    lines_ov.append('STREAM OVERLAP TOP30 AUDIT — NOT A FINAL MEMBER SELECTOR')
    lines_ov.append(f'PLAY_DATE: {play_date}')
    lines_ov.append(f'HISTORY_THROUGH: {history_through}')
    lines_ov.append('Rule: take Top 30 from each lane, group by stream, keep all lane-qualified core/member rows for overlapping streams.')
    lines_ov.append('')
    lines_ov.append('Rank | Stream | Lanes | BestLaneRank | Rows | Core/Member Options')
    lines_ov.append('-'*120)
    if stream_overlap_streams.empty:
        lines_ov.append('NO STREAM OVERLAP ROWS PRODUCED.')
    else:
        for _, rr in stream_overlap_streams.iterrows():
            lines_ov.append(f"{int(rr.get('StreamOverlapRank',0)):02d} | {rr.get('stream','')} | {int(rr.get('appears_in_lanes_top30',0))} | {int(rr.get('best_lane_rank_for_stream',0))} | {int(rr.get('rows_for_stream_top30',0))} | {rr.get('core_member_options_top30','')}")
    (out/'06_STREAM_OVERLAP_TOP30_PRINTABLE.txt').write_text('\n'.join(lines_ov), encoding='utf-8')



    # Decision-layer strategy audit: keeps all 6 lanes, compares practical candidate rules.
    # No future winners are used here. These files are for comparing row cost and reviewing candidates.
    def _dedupe_rank(df_in, sort_cols=None, asc=None, cap=None):
        if df_in is None or df_in.empty:
            return pd.DataFrame()
        d = _normalize_play_cols(df_in.copy())
        if 'lane_rank' in d.columns:
            d['_lane_rank_num'] = pd.to_numeric(d['lane_rank'], errors='coerce').fillna(9999)
        else:
            d['_lane_rank_num'] = 9999
        if '_stream_lane_count_top30' not in d.columns and 'stream' in d.columns and not stream_overlap_streams.empty:
            d['_stream_lane_count_top30'] = d['stream'].map(stream_overlap_streams.set_index('stream')['appears_in_lanes_top30'].to_dict()).fillna(1)
        if '_stream_overlap_rank' not in d.columns and 'stream' in d.columns and not stream_overlap_streams.empty:
            d['_stream_overlap_rank'] = d['stream'].map(stream_overlap_streams.set_index('stream')['StreamOverlapRank'].to_dict()).fillna(9999)
        if 'final_plus_transition_x15' in d.columns:
            d['_x15_num'] = pd.to_numeric(d['final_plus_transition_x15'], errors='coerce').fillna(0)
        else:
            d['_x15_num'] = 0
        default_sort = ['_stream_lane_count_top30','_stream_overlap_rank','_lane_rank_num','_x15_num','stream','core','member']
        default_asc = [False, True, True, False, True, True, True]
        d = d.sort_values(sort_cols or default_sort, ascending=asc or default_asc)
        keys=[c for c in ['stream','seed','core','member'] if c in d.columns]
        if keys:
            d = d.drop_duplicates(keys, keep='first')
        if cap is not None:
            d = d.head(int(cap)).copy()
        d.insert(0, 'DecisionRank', range(1, len(d)+1))
        return d

    decision_dir = out / '07_DECISION_LAYER_STRATEGIES'
    decision_dir.mkdir(exist_ok=True)
    strategy_frames = {}
    top30_rows = pd.DataFrame()
    if not lane_all.empty:
        top30_rows = lane_all[pd.to_numeric(lane_all.get('lane_rank', 9999), errors='coerce').fillna(9999).le(30)].copy()
        if not top30_rows.empty and not stream_overlap_streams.empty:
            top30_rows['_stream_lane_count_top30'] = top30_rows['stream'].map(stream_overlap_streams.set_index('stream')['appears_in_lanes_top30'].to_dict()).fillna(1)
            top30_rows['_stream_overlap_rank'] = top30_rows['stream'].map(stream_overlap_streams.set_index('stream')['StreamOverlapRank'].to_dict()).fillna(9999)

    # A: all lane Top30 rows, no stream deletion; cap after dedupe.
    strategy_frames['A_ALL_LANE_TOP30_ROWS_CAP50'] = _dedupe_rank(top30_rows, cap=play_cap)

    # B: exact Top30 rows whose stream appears in 2+ lanes.
    if not top30_rows.empty:
        strategy_frames['B_ONLY_STREAMS_OVERLAP_2PLUS_TOP30_ROWS'] = _dedupe_rank(top30_rows[top30_rows['_stream_lane_count_top30'].ge(2)], cap=play_cap)
    else:
        strategy_frames['B_ONLY_STREAMS_OVERLAP_2PLUS_TOP30_ROWS'] = pd.DataFrame()

    # C: streams with overlap 2+, but keep every lane-qualified member for those streams before cap.
    if not lane_all.empty and not stream_overlap_streams.empty:
        ov2_streams = set(stream_overlap_streams.loc[stream_overlap_streams['appears_in_lanes_top30'].ge(2), 'stream'].astype(str))
        c_rows = lane_all[lane_all['stream'].astype(str).isin(ov2_streams)].copy()
        strategy_frames['C_OVERLAP_2PLUS_STREAMS_KEEP_ALL_LANE_MEMBERS'] = _dedupe_rank(c_rows, cap=play_cap)
    else:
        strategy_frames['C_OVERLAP_2PLUS_STREAMS_KEEP_ALL_LANE_MEMBERS'] = pd.DataFrame()

    # D: priority ladder: overlap3 streams first, overlap2 streams next, single-lane best ranks last.
    if not top30_rows.empty:
        drows = top30_rows.copy()
        cnt = pd.to_numeric(drows.get('_stream_lane_count_top30', 1), errors='coerce').fillna(1)
        drows['_decision_priority_group'] = np.where(cnt.ge(3), 1, np.where(cnt.ge(2), 2, 3))
        strategy_frames['D_OVERLAP3_THEN_2_THEN_SINGLE_BEST_RANK_CAP50'] = _dedupe_rank(
            drows,
            sort_cols=['_decision_priority_group','_stream_overlap_rank','_lane_rank_num','_x15_num','stream','core','member'],
            asc=[True, True, True, False, True, True, True],
            cap=play_cap
        )
    else:
        strategy_frames['D_OVERLAP3_THEN_2_THEN_SINGLE_BEST_RANK_CAP50'] = pd.DataFrame()

    # E: one best row per overlap-2+ stream.
    if not top30_rows.empty:
        ebase = top30_rows[top30_rows['_stream_lane_count_top30'].ge(2)].copy()
        eranked = _dedupe_rank(ebase, cap=None)
        if not eranked.empty and 'stream' in eranked.columns:
            eranked = eranked.sort_values(['stream','DecisionRank']).drop_duplicates(['stream'], keep='first')
            eranked = eranked.sort_values(['_stream_lane_count_top30','_stream_overlap_rank','_lane_rank_num'], ascending=[False, True, True]).head(int(play_cap)).copy()
            eranked['DecisionRank'] = range(1, len(eranked)+1)
        strategy_frames['E_ONE_BEST_ROW_PER_OVERLAP_2PLUS_STREAM'] = eranked
    else:
        strategy_frames['E_ONE_BEST_ROW_PER_OVERLAP_2PLUS_STREAM'] = pd.DataFrame()

    # F: current merged-by-best-lane-rank cap list (renamed candidate file).
    strategy_frames['F_MERGED_BY_BEST_LANE_RANK_CAP50'] = final.copy()
    if not strategy_frames['F_MERGED_BY_BEST_LANE_RANK_CAP50'].empty:
        if 'DecisionRank' not in strategy_frames['F_MERGED_BY_BEST_LANE_RANK_CAP50'].columns:
            strategy_frames['F_MERGED_BY_BEST_LANE_RANK_CAP50'].insert(0, 'DecisionRank', range(1, len(strategy_frames['F_MERGED_BY_BEST_LANE_RANK_CAP50'])+1))

    strategy_summary_rows=[]
    for sid, sdf in strategy_frames.items():
        if sdf is None or sdf.empty:
            sdf = pd.DataFrame()
        safe = ''.join(ch if ch.isalnum() or ch in ['_','-'] else '_' for ch in sid)
        sdf.to_csv(decision_dir / f'{safe}.csv', index=False)
        streams = int(sdf['stream'].astype(str).nunique()) if 'stream' in sdf.columns and not sdf.empty else 0
        rows_count = int(len(sdf))
        avg_rank = float(pd.to_numeric(sdf.get('lane_rank', pd.Series(dtype=float)), errors='coerce').dropna().mean()) if not sdf.empty and 'lane_rank' in sdf.columns else np.nan
        max_stream_overlap = int(pd.to_numeric(sdf.get('_stream_lane_count_top30', pd.Series(dtype=float)), errors='coerce').fillna(0).max()) if not sdf.empty and '_stream_lane_count_top30' in sdf.columns else np.nan
        strategy_summary_rows.append({
            'strategy_id': sid,
            'rows': rows_count,
            'unique_streams': streams,
            'avg_lane_rank_if_available': avg_rank,
            'max_stream_overlap_top30_if_available': max_stream_overlap,
            'description': {
                'A_ALL_LANE_TOP30_ROWS_CAP50':'All rows ranked Top30 inside any lane; dedupe exact stream/seed/core/member; cap at 50.',
                'B_ONLY_STREAMS_OVERLAP_2PLUS_TOP30_ROWS':'Only Top30 rows where the stream appears in 2+ lane Top30 lists.',
                'C_OVERLAP_2PLUS_STREAMS_KEEP_ALL_LANE_MEMBERS':'Find streams with Top30 overlap 2+, then keep all lane-qualified members for those streams before cap.',
                'D_OVERLAP3_THEN_2_THEN_SINGLE_BEST_RANK_CAP50':'Priority ladder: overlap 3 streams first, overlap 2 next, then single-lane best-ranked rows to fill cap.',
                'E_ONE_BEST_ROW_PER_OVERLAP_2PLUS_STREAM':'One strongest row per stream among streams that appear in 2+ lane Top30 lists.',
                'F_MERGED_BY_BEST_LANE_RANK_CAP50':'The renamed merged lane candidate list ranked by best lane rank.'
            }.get(sid,'')
        })
    decision_summary = pd.DataFrame(strategy_summary_rows)
    decision_summary.to_csv(out/'07_DECISION_LAYER_STRATEGY_SUMMARY.csv', index=False)

    # Printable decision-layer summary.
    lines_dec = []
    lines_dec.append('DECISION LAYER STRATEGY AUDIT — KEEPS ALL 6 LANES')
    lines_dec.append(f'PLAY_DATE: {play_date}')
    lines_dec.append(f'HISTORY_THROUGH: {history_through}')
    lines_dec.append('Purpose: compare candidate play strategies without deleting any lane from the app.')
    lines_dec.append('')
    lines_dec.append('Strategy | Rows | Streams | Description')
    lines_dec.append('-'*140)
    for _, rr in decision_summary.iterrows():
        lines_dec.append(f"{rr.get('strategy_id','')} | {int(rr.get('rows',0))} | {int(rr.get('unique_streams',0))} | {rr.get('description','')}")
    (out/'07_DECISION_LAYER_STRATEGY_SUMMARY.txt').write_text('\n'.join(lines_dec), encoding='utf-8')

    # Printable TXT.
    lines=[]
    lines.append(f'EMERGENCY DAILY PLAYER V1.4 — MERGED LANE CANDIDATES')
    lines.append(f'PLAY_DATE: {play_date}')
    lines.append(f'HISTORY_THROUGH: {history_through}')
    lines.append(f'ROWS: {len(final)} / CAP {int(play_cap)}')
    lines.append('')
    lines.append('Rank | Stream | Seed | Core | Member | Lanes | BestLaneRank | Score')
    lines.append('-'*96)
    if final.empty:
        lines.append('NO PLAYABLE ROWS PRODUCED BY CURRENT DEFAULT LANES.')
    else:
        for _,r in final.iterrows():
            lines.append(f"{int(r['FinalRank']):02d} | {r.get('stream','')} | {str(r.get('seed','')).zfill(4)} | {str(r.get('core','')).zfill(3)} | {str(r.get('member','')).zfill(4)} | {int(r.get('lane_count',0))} | {int(r.get('best_lane_rank',0))} | {float(r.get('consensus_score',0)):.2f}")
    txt='\n'.join(lines)
    (out/'02_MERGED_LANE_CANDIDATES_PRINTABLE.txt').write_text(txt, encoding='utf-8')

    zip_path = out.parent / f'{EMERGENCY_BUILD_ID}_{play_date}_OUTPUTS.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(out))
    return {'summary': run_summary, 'lane_summary': lane_summary_df, 'merged_candidates': final, 'final': final, 'lane_rows': lane_all, 'stream_overlap_streams': stream_overlap_streams, 'stream_overlap_rows': stream_overlap_rows, 'step2': step2, 'out_dir': str(out), 'zip_path': str(zip_path)}
