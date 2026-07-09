
#!/usr/bin/env python3
"""
DL_V16_STEP2_BUCKET_WORKBENCH

This is the corrected reproducible daily ladder builder.
It does NOT require prebuilt Step 2 files. It builds them from history + bundled profiles.

Locked meanings:
- Step 0: load history/seeds, optional AZ/MD exclusion.
- Step 1: full enumeration of playable streams x FULL120 AABC cores/members, watched-8 flagged.
- Stream Gate Layer: locked layer after Step 1, not called Step 1. DEFAULT branch now uses baseline N47 prior-180 watched-8 frequency/cadence gate because that is the known 06/19 baseline reproduction gate. Winner-scale/cadence tie-breaker is still exported as audit/research.
- Q2 Base Contenders: top2 watched cores per stream, top1 member per surviving stream-core.
- Step 2: corrected seed->member transition overlay over Q2/base contender rows, x15, no hard deletion.
- Step 3: mean/median/average row-count tests over both cores and streams, including Top47 and Top50 rescue checks. No final gate lock here.
- Mirror hard filtering is not used.
"""
from __future__ import annotations
from pathlib import Path
from itertools import combinations
import argparse, json, math, re, zipfile
import numpy as np
import pandas as pd

BUILD = "DL_V16_STEP2_BUCKET_WORKBENCH"
WATCHED8 = {"027","067","138","145","389","457","567","679"}
NONPLAYABLE_MARKERS = ("Arizona |", "Maryland |")

# ---------- normalization ----------
def digits(x):
    if pd.isna(x): return ""
    return re.sub(r"\D", "", str(x).replace(".0", ""))

def norm4(x):
    s=digits(x)
    return s.zfill(4)[-4:] if s else ""

def boxed_member(x):
    s=norm4(x)
    return "".join(sorted(s)) if s else ""

def core_from_result(x):
    s=norm4(x)
    if not s: return ""
    vals=sorted(set(s))
    return "".join(vals) if len(vals)==3 else ""

def structure4(x):
    s=norm4(x)
    if not s: return ""
    counts=tuple(sorted([s.count(d) for d in set(s)], reverse=True))
    return {(4,):"AAAA", (3,1):"AAAB", (2,2):"AABB", (2,1,1):"AABC", (1,1,1,1):"ABCD"}.get(counts,"OTHER")

def parity_pattern(s):
    s=norm4(s)
    return "".join("E" if int(c)%2==0 else "O" for c in s) if s else ""

def parity_count(s):
    s=norm4(s)
    if not s: return ""
    e=sum(int(c)%2==0 for c in s)
    return f"{e}E{4-e}O"

def highlow(s):
    s=norm4(s)
    return "".join("L" if int(c)<=4 else "H" for c in s) if s else ""

def spread(s):
    s=norm4(s)
    return max(map(int,s))-min(map(int,s)) if s else np.nan

def digsum(s):
    s=norm4(s)
    return sum(map(int,s)) if s else np.nan

def sum_bucket(v):
    try: v=int(v)
    except Exception: return ""
    if v<=9: return "sum_00_09"
    if v<=14: return "sum_10_14"
    if v<=18: return "sum_15_18"
    if v<=22: return "sum_19_22"
    if v<=27: return "sum_23_27"
    return "sum_28_36"

def gap_bucket(n):
    try: n=int(n)
    except Exception: return "gap_unknown"
    if n<=7: return "gap_000_007"
    if n<=14: return "gap_008_014"
    if n<=30: return "gap_015_030"
    if n<=60: return "gap_031_060"
    if n<=120: return "gap_061_120"
    return "gap_121_plus"

def core_members(core):
    c=str(core).zfill(3)[-3:]
    ds=sorted(c)
    return sorted({"".join(sorted(ds+[d])) for d in ds})

def all_aabc_members_and_cores():
    rows=[]
    for comb in combinations('0123456789',3):
        c=''.join(comb)
        for m in core_members(c):
            rows.append({'core':c,'member':m})
    return pd.DataFrame(rows)

CORE_MEMBER_TABLE=all_aabc_members_and_cores()
CORE120=sorted(CORE_MEMBER_TABLE['core'].unique())


# ---------- flexible TXT/CSV input ----------
def read_table_any(path):
    p=Path(path)
    # Try normal CSV first, then delimiter inference. Keeps leading zeros as strings.
    try:
        return pd.read_csv(p, dtype=str)
    except Exception:
        pass
    try:
        return pd.read_csv(p, dtype=str, sep=None, engine='python')
    except Exception:
        pass
    # Last-resort simple text parser: lines like Date State Game Result or Stream Result.
    lines=p.read_text(errors='replace').splitlines()
    rows=[]
    for line in lines:
        raw=line.strip()
        if not raw or raw.startswith('#'):
            continue
        parts=re.split(r'[,\t|]+', raw)
        parts=[x.strip() for x in parts if x.strip()]
        if len(parts)>=4 and re.search(r'\d{4}', parts[-1]):
            rows.append({'Date':parts[0], 'State':parts[1], 'Game':' | '.join(parts[2:-1]), 'Result4':norm4(parts[-1])})
        elif len(parts)>=2 and re.search(r'\d{4}', parts[-1]):
            rows.append({'StreamKey':' | '.join(parts[:-1]), 'Result4':norm4(parts[-1])})
    if rows:
        return pd.DataFrame(rows)
    raise ValueError(f'Could not parse input file as CSV/TXT: {p}')

# ---------- history ----------
def read_history(path):
    df=read_table_any(path)
    colmap={}
    # choose one result/base4 column. Prefer Result4/base4 over decorative Result.
    cols_lower={c.lower().strip(): c for c in df.columns}
    result_col=None
    for candidate in ['result4','base4','winning_number','result']:
        if candidate in cols_lower:
            result_col=cols_lower[candidate]; break
    for c in df.columns:
        lc=c.lower().strip()
        if lc in ['date','draw_date','drawdate']: colmap[c]='draw_date'
        elif lc in ['streamkey','stream','stream_name','streamname']: colmap[c]='stream'
        elif c == result_col: colmap[c]='base4'
        elif lc=='state': colmap[c]='state'
        elif lc=='game': colmap[c]='game'
    df=df.rename(columns=colmap)
    # drop duplicate columns caused by files that include both Result and Result4
    df=df.loc[:, ~df.columns.duplicated()].copy()
    if 'stream' not in df.columns and {'state','game'}.issubset(df.columns):
        df['stream']=df['state'].astype(str).str.strip()+" | "+df['game'].astype(str).str.strip()
    missing=[c for c in ['draw_date','stream','base4'] if c not in df.columns]
    if missing: raise ValueError(f"History missing columns: {missing}")
    df=df.copy()
    df['draw_date']=pd.to_datetime(df['draw_date'], errors='coerce').dt.strftime('%Y-%m-%d')
    df['base4']=df['base4'].map(norm4)
    df['member']=df['base4'].map(boxed_member)
    df['core']=df['base4'].map(core_from_result)
    df['structure']=df['base4'].map(structure4)
    df['stream']=df['stream'].astype(str).str.strip()
    df=df[df['draw_date'].notna() & df['base4'].ne('') & df['stream'].ne('')].copy()
    df=df.drop_duplicates(['draw_date','stream','base4']).sort_values(['draw_date','stream']).reset_index(drop=True)
    return df

def apply_exclusions(hist, exclude_az_md=True):
    if not exclude_az_md: return hist.copy(), pd.DataFrame([{'excluded_stream_marker':'none','rows_removed':0,'streams_removed':0}])
    mask=hist['stream'].str.startswith(NONPLAYABLE_MARKERS)
    audit=[]
    for marker in NONPLAYABLE_MARKERS:
        m=hist['stream'].str.startswith(marker)
        audit.append({'excluded_stream_marker':marker, 'rows_removed':int(m.sum()), 'streams_removed':int(hist.loc[m,'stream'].nunique())})
    return hist.loc[~mask].copy(), pd.DataFrame(audit)

def derive_seed_events(hist, history_through, play_date=None):
    ht=pd.to_datetime(history_through).strftime('%Y-%m-%d')
    if play_date is None:
        play_date=(pd.to_datetime(ht)+pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    else: play_date=pd.to_datetime(play_date).strftime('%Y-%m-%d')
    h=hist[hist['draw_date'].le(ht)].sort_values(['stream','draw_date']).copy()
    latest=h.groupby('stream', as_index=False).tail(1).copy()
    events=[]
    for _,r in latest.iterrows():
        seed=r['base4']
        s=r['stream']
        prior_aabc=h[(h['stream'].eq(s)) & (h['core'].ne(''))].tail(3)['core'].tolist()[::-1]
        events.append({
            'play_date':play_date,'history_through':ht,'stream':s,
            'prior_draw_date':r['draw_date'],'seed':seed,'seed_member':boxed_member(seed),'seed_core':core_from_result(seed),
            'seed_sum':digsum(seed),'seed_sum_bucket':sum_bucket(digsum(seed)),
            'seed_spread':spread(seed),'seed_parity':parity_pattern(seed),'seed_parity_count':parity_count(seed),
            'seed_highlow':highlow(seed),'seed_structure':structure4(seed),
            'last1_aabc_core':prior_aabc[0] if len(prior_aabc)>0 else '',
            'last2_aabc_core':prior_aabc[1] if len(prior_aabc)>1 else '',
            'last3_aabc_core':prior_aabc[2] if len(prior_aabc)>2 else '',
        })
    return pd.DataFrame(events)

# ---------- profiles ----------
def read_csv_if_exists(path):
    p=Path(path)
    return pd.read_csv(p, dtype=str) if p.exists() else pd.DataFrame()

def resolve_profile_file(profile_dir, filename):
    """Find profile CSVs whether GitHub kept profiles/ or flattened the CSVs to repo root.
    Search order is explicit profiles/ first, then repo root/current working directory, then this code file's folder.
    """
    candidates = [
        Path(profile_dir) / filename,
        Path.cwd() / filename,
        Path(__file__).resolve().parent / filename,
        Path(__file__).resolve().parent / 'profiles' / filename,
    ]
    seen=set()
    for c in candidates:
        try:
            key=str(c.resolve())
        except Exception:
            key=str(c)
        if key in seen:
            continue
        seen.add(key)
        if c.exists():
            return c
    return Path(profile_dir) / filename

def load_profiles(profile_dir):
    p=Path(profile_dir)
    prof={}
    for key,name in {
        'stream_core':'V6_8CORE_USABLE_STREAM_CORE_SIGNALS.csv',
        'seed_trait':'V6_8CORE_USABLE_SEED_TRAIT_SIGNALS.csv',
        'stream_seed_trait':'V6_8CORE_USABLE_STREAM_SEED_TRAIT_SIGNALS.csv',
        'member_role':'V6_8CORE_MEMBER_ROLE_PROFILES.csv',
        'exact_member':'V6_8CORE_EXACT_STREAM_CORE_MEMBER_PROFILES.csv',
        'cadence':'V6_8CORE_CADENCE_SIGNALS.csv',
    }.items(): prof[key]=read_csv_if_exists(resolve_profile_file(p,name))
    for df in prof.values():
        if not df.empty:
            for c in df.columns:
                if c.endswith('score') or c in ['confidence_score','exact_pair_score','member_share_pct','relative_lift_x','hit_count','sample_size']:
                    df[c]=pd.to_numeric(df[c], errors='coerce')
            for c in ['core_str','member_str']:
                if c in df.columns: df[c]=df[c].astype(str).str.replace(r'\.0$','',regex=True).str.zfill(3 if c=='core_str' else 4)
    return prof

def profile_lookup_maps(prof):
    maps={}
    sc=prof.get('stream_core',pd.DataFrame())
    maps['stream_core']={(str(r.StreamKey), str(r.core_str).zfill(3)): float(r.confidence_score or 0) for _,r in sc.iterrows()} if not sc.empty else {}
    ss=prof.get('stream_seed_trait',pd.DataFrame())
    maps['stream_seed']={(str(r.StreamKey), str(r.core_str).zfill(3), str(r.trait_name), str(r.trait_value)): float(r.confidence_score or 0) for _,r in ss.iterrows()} if not ss.empty else {}
    st=prof.get('seed_trait',pd.DataFrame())
    maps['seed_trait']={(str(r.core_str).zfill(3), str(r.trait_name), str(r.trait_value)): float(r.confidence_score or 0) for _,r in st.iterrows()} if not st.empty else {}
    cad=prof.get('cadence',pd.DataFrame())
    maps['cadence']={(str(r.core_str).zfill(3), str(r.SameCoreGapBucket)): float(r.confidence_score or 0) for _,r in cad.iterrows()} if not cad.empty else {}
    mr=prof.get('member_role',pd.DataFrame())
    maps['member_role']={(str(r.core_str).zfill(3), str(r.member_str).zfill(4)): float(r.member_share_pct or 0)/100.0 for _,r in mr.iterrows()} if not mr.empty else {}
    ex=prof.get('exact_member',pd.DataFrame())
    maps['exact']={(str(r.StreamKey), str(r.core_str).zfill(3), str(r.member_str).zfill(4)): float(r.exact_pair_score or 0) for _,r in ex.iterrows()} if not ex.empty else {}
    return maps

def days_since_core(hist, stream, core, through):
    h=hist[(hist['draw_date'].le(through)) & (hist['stream'].eq(stream)) & (hist['core'].eq(core))]
    if h.empty: return 9999
    return int((pd.to_datetime(through)-pd.to_datetime(h['draw_date'].max())).days)

# ---------- Step1 full enumeration/scoring ----------
def build_gap_map(hist, events):
    # Precompute same-core gap by stream/core once. Avoid filtering the history for every candidate row.
    through=str(events['history_through'].iloc[0]) if len(events) else str(hist['draw_date'].max())
    h=hist[(hist['draw_date'].le(through)) & (hist['core'].ne(''))].copy()
    last=h.sort_values('draw_date').groupby(['stream','core'], as_index=False).tail(1)[['stream','core','draw_date']]
    last['days_since']=(pd.to_datetime(through)-pd.to_datetime(last['draw_date'])).dt.days.astype(int)
    return {(r.stream, r.core): gap_bucket(r.days_since) for _,r in last.iterrows()}

def build_full_enumeration(hist, events, prof, include_full120=True):
    maps=profile_lookup_maps(prof)
    gap_map=build_gap_map(hist, events)
    cm_records=CORE_MEMBER_TABLE.to_dict('records')
    rows=[]
    for ev in events.to_dict('records'):
        trait_vals={
            'seed_parity': ev['seed_parity'],
            'seed_highlow': ev['seed_highlow'],
            'seed_structure': ev['seed_structure'],
            'seed_parity_count': ev['seed_parity_count'],
            'seed_sum_bucket': ev['seed_sum_bucket'],
        }
        # Core-level scores are repeated for the three members of each core. Cache them per stream/core.
        core_cache={}
        for cm in cm_records:
            core=cm['core']; member=cm['member']
            if core not in core_cache:
                score_stream_core=maps['stream_core'].get((ev['stream'], core), 0.0)
                score_seed_trait=sum(maps['seed_trait'].get((core,k,v),0.0) for k,v in trait_vals.items())
                score_stream_seed_trait=sum(maps['stream_seed'].get((ev['stream'],core,k,v),0.0) for k,v in trait_vals.items())
                gap=gap_map.get((ev['stream'], core), 'gap_121_plus')
                score_cadence=maps['cadence'].get((core,gap),0.0)
                core_score=(score_stream_core*100) + (score_seed_trait*60) + (score_stream_seed_trait*125) + (score_cadence*40)
                core_cache[core]=(score_stream_core,score_seed_trait,score_stream_seed_trait,score_cadence,gap,core_score)
            score_stream_core,score_seed_trait,score_stream_seed_trait,score_cadence,gap,core_score=core_cache[core]
            score_member_role=maps['member_role'].get((core,member),0.0)
            score_exact=maps['exact'].get((ev['stream'],core,member),0.0)
            member_score=core_score + (score_member_role*20) + score_exact
            all_support=sum(float(x)>0 for x in [score_stream_core,score_seed_trait,score_stream_seed_trait,score_cadence,score_member_role,score_exact])
            major_support=sum(float(x)>0 for x in [score_stream_core,score_seed_trait,score_stream_seed_trait,score_cadence,score_exact])
            rows.append({
                **ev,'core':core,'member':member,'is_watched8_core':core in WATCHED8,
                'score_stream_core_usable':score_stream_core,
                'score_seed_trait_usable':score_seed_trait,
                'score_stream_seed_trait_usable':score_stream_seed_trait,
                'score_cadence':score_cadence,
                'score_member_role':score_member_role,
                'score_exact_stream_core_member':score_exact,
                'same_core_gap_bucket':gap,
                'profile_core_score':core_score,
                'profile_final_member_score':member_score,
                'major_support_count':major_support,'all_support_count':all_support,
                'mirror_filter_used':False,
            })
    df=pd.DataFrame(rows)
    df['core_rank_in_stream']=df.groupby(['play_date','stream'])['profile_core_score'].rank(method='first', ascending=False).astype(int)
    df['member_rank_in_stream_core']=df.groupby(['play_date','stream','core'])['profile_final_member_score'].rank(method='first', ascending=False).astype(int)
    return df

# ---------- Stream Gate Layer ----------
def _watched8_hit_windows(hist, through):
    """Build stream-level watched-8 AABC history features through date.
    This reproduces the baseline-style N47 stream gate family:
    prior-window watched-8 hit counts + cadence pressure. It is deliberately
    separate from canonical Step 1 full enumeration.
    """
    through=pd.to_datetime(through)
    h=hist[hist['draw_date'].le(through.strftime('%Y-%m-%d'))].copy()
    streams=sorted(h['stream'].dropna().unique())
    rows=[]
    for st in streams:
        hs=h[h['stream'].eq(st)].copy()
        w=hs[hs['core'].isin(WATCHED8)].copy()
        dts=pd.to_datetime(w['draw_date'], errors='coerce') if not w.empty else pd.Series([], dtype='datetime64[ns]')
        rec={'stream':st}
        for n in [7,14,30,60,90,180]:
            rec[f'hits{n}']=float((dts >= (through-pd.Timedelta(days=n-1))).sum()) if len(dts) else 0.0
        if not w.empty:
            last=pd.to_datetime(w['draw_date'].max())
            days_since=int((through-last).days)
        else:
            days_since=9999
        hits180=rec['hits180']
        expected_gap=180.0/(hits180+1.0)
        cadence_urgency=max(0.0, math.log1p(days_since/expected_gap)) if days_since<9999 else 0.0
        # This cadence score matches the older watchlist scale closely enough to
        # recreate the 06/19 N47 inclusion behavior, while remaining daily-generatable.
        cadence_score=(rec['hits7']*8.0 + rec['hits14']*4.0 + rec['hits30']*2.0 +
                       rec['hits60']*1.0 + rec['hits90']*0.75 + rec['hits180']*0.25 + cadence_urgency)
        if rec['hits60']>=4 or rec['hits30']>=3 or rec['hits14']>=2:
            bucket='ACTIVE'
        elif rec['hits180']>=6 or cadence_urgency>=0.5:
            bucket='WATCH'
        else:
            bucket='LOW'
        rec.update({'days_since_last':days_since,'expected_gap':expected_gap,
                    'cadence_urgency':cadence_urgency,'cadence_score':cadence_score,
                    'cadence_bucket':bucket})
        # Baseline/repro score: prior-180 coverage is primary; shorter windows and cadence break ties.
        rec['baseline_n47_score']=(rec['hits180']*100.0 + rec['hits90']*10.0 + rec['hits60']*3.0 +
                                   rec['hits30']*2.0 + rec['hits14']*0.5 + rec['hits7']*0.25 +
                                   rec['cadence_score'])
        rows.append(rec)
    return pd.DataFrame(rows)

def build_winner_scale_gate(full):
    # winner scale = strongest watched8 evidence in each stream; cadence = max cadence among watched8 rows.
    w=full[full['is_watched8_core']].copy()
    g=w.groupby(['play_date','stream'], as_index=False).agg(
        winner_scale_score=('profile_core_score','max'),
        cadence_score=('score_cadence','max'),
        watched8_core_evidence_rows=('major_support_count', lambda x:int((x>0).sum()))
    )
    for c in ['winner_scale_score','cadence_score']:
        mu=g[c].mean(); sd=g[c].std(ddof=0) or 1.0
        g['z_'+c]=(g[c]-mu)/sd
    g['loser_scale_score']=0.0
    g['winner_scale_cadence_final_score']=g['z_winner_scale_score'] + 0.1*g['z_cadence_score']
    g=g.sort_values(['play_date','winner_scale_cadence_final_score','winner_scale_score','cadence_score','stream'], ascending=[True,False,False,False,True]).copy()
    g['winner_scale_gate_rank']=g.groupby('play_date').cumcount()+1
    g['in_winner_scale_top47']=g['winner_scale_gate_rank'].le(47)
    g['in_winner_scale_top50']=g['winner_scale_gate_rank'].le(50)
    return g

def build_stream_gate(full, hist=None, history_through=None, mode='baseline_n47'):
    """Build active stream gate and export audit columns.

    Active default is baseline_n47, because the known 06/19 ladder used the
    N47 prior-180 watched-8 stream gate and Oregon 10pm must survive before Step 3.
    Winner-scale/cadence tie-breaker is still calculated and merged for audit.
    """
    play_date=str(full['play_date'].iloc[0]) if len(full) else ''
    audit=build_winner_scale_gate(full)
    if hist is None or history_through is None:
        # Fallback to winner-scale if history was not passed, but normal daily runs pass history.
        g=audit.rename(columns={'winner_scale_gate_rank':'stream_gate_rank',
                                'winner_scale_cadence_final_score':'stream_gate_final_score'}).copy()
        g['active_stream_gate_mode']='winner_scale_cadence_fallback'
    else:
        base=_watched8_hit_windows(hist, history_through)
        base['play_date']=play_date
        base=base.sort_values(['baseline_n47_score','hits180','hits90','hits60','cadence_score','stream'], ascending=[False,False,False,False,False,True]).copy()
        base['baseline_n47_rank']=range(1,len(base)+1)
        base['in_baseline_n47_top47']=base['baseline_n47_rank'].le(47)
        base['in_baseline_n47_top50']=base['baseline_n47_rank'].le(50)
        g=base.merge(audit,on=['play_date','stream'],how='left')
        if mode=='winner_scale_cadence':
            g['stream_gate_rank']=g['winner_scale_gate_rank']
            g['stream_gate_final_score']=g['winner_scale_cadence_final_score']
            g['active_stream_gate_mode']='winner_scale_cadence_tiebreaker'
        else:
            g['stream_gate_rank']=g['baseline_n47_rank']
            g['stream_gate_final_score']=g['baseline_n47_score']
            g['active_stream_gate_mode']='baseline_n47_prior180_frequency_cadence'
        g=g.sort_values(['play_date','stream_gate_rank','stream']).copy()
    g['in_stream_gate_top47']=g['stream_gate_rank'].le(47)
    g['in_stream_gate_top50']=g['stream_gate_rank'].le(50)
    return g

# ---------- Base Q2 contenders ----------
def _add_q2_repair_ranks(w):
    """Add core/member ranks used by Q2 repair modes."""
    agg=w.groupby(['play_date','stream','core'], as_index=False).agg(
        core_score=('profile_core_score','max'),
        max_member_score=('profile_final_member_score','max'),
        max_major=('major_support_count','max'),
        max_all=('all_support_count','max'),
        max_exact=('score_exact_stream_core_member','max')
    )
    agg=agg.sort_values(['play_date','stream','core_score','max_major','max_all','max_member_score','core'], ascending=[True,True,False,False,False,False,True])
    agg['q2_rank_core_score']=agg.groupby(['play_date','stream']).cumcount()+1
    agg=agg.sort_values(['play_date','stream','max_member_score','max_major','max_all','core_score','core'], ascending=[True,True,False,False,False,False,True])
    agg['q2_rank_membermax_score']=agg.groupby(['play_date','stream']).cumcount()+1
    agg=agg.sort_values(['play_date','stream','max_major','max_all','max_member_score','core'], ascending=[True,True,False,False,False,True])
    agg['q2_rank_support']=agg.groupby(['play_date','stream']).cumcount()+1
    out=w.merge(agg[['play_date','stream','core','q2_rank_core_score','q2_rank_membermax_score','q2_rank_support','max_major','max_all']], on=['play_date','stream','core'], how='left')
    out=out.sort_values(['play_date','stream','core','profile_final_member_score','major_support_count','all_support_count','member'], ascending=[True,True,True,False,False,False,True]).copy()
    out['q2_rank_member_final']=out.groupby(['play_date','stream','core']).cumcount()+1
    return out

def build_q2_contenders(full, stream_gate, use_top=47, q2_mode='balanced_repair'):
    """Build base contender rows for Step 2.

    Modes:
    - baseline_94: original Q2_CORETOP2_MEMTOP1, top2 cores by core score and top1 member.
    - small_support_149: top2 support cores and top1 member + major>=4 member rescue.
    - balanced_repair: top2 core-score plus top5 major3/all4 core rescue; top1 member plus rank2 major3/all4 rescue.
    - wide_repair: top2 core-score plus any major3/all4 core rescue; top1 member plus rank2 major3/all4 rescue.

    The default is balanced_repair because V2 was losing too many winners before Step 3.
    Mirror hard filtering is not used.
    """
    gate_col=f'in_stream_gate_top{use_top}'
    streams=set(stream_gate.loc[stream_gate[gate_col], 'stream'])
    w=full[(full['is_watched8_core']) & (full['stream'].isin(streams))].copy()
    w=_add_q2_repair_ranks(w)

    if q2_mode == 'baseline_94':
        core_keep=w['q2_rank_core_score'].le(2)
        member_keep=w['q2_rank_member_final'].le(1)
    elif q2_mode == 'small_support_149':
        core_keep=w['q2_rank_support'].le(2)
        member_keep=w['q2_rank_member_final'].le(1) | w['major_support_count'].ge(4)
    elif q2_mode == 'wide_repair':
        core_keep=w['q2_rank_core_score'].le(2) | ((w['max_major'].ge(3)) & (w['max_all'].ge(4)))
        member_keep=w['q2_rank_member_final'].le(1) | ((w['q2_rank_member_final'].le(2)) & (w['major_support_count'].ge(3)) & (w['all_support_count'].ge(4)))
    elif q2_mode == 'balanced_repair':
        core_keep=w['q2_rank_core_score'].le(2) | ((w['q2_rank_core_score'].le(5)) & (w['max_major'].ge(3)) & (w['max_all'].ge(4)))
        member_keep=w['q2_rank_member_final'].le(1) | ((w['q2_rank_member_final'].le(2)) & (w['major_support_count'].ge(3)) & (w['all_support_count'].ge(4)))
    else:
        raise ValueError(f'Unknown q2_mode: {q2_mode}')

    q=w[core_keep & member_keep].copy()
    q['q2_mode']=q2_mode
    q['q2_core_rank_in_stream']=q['q2_rank_core_score']
    q['q2_member_rank_in_stream_core']=q['q2_rank_member_final']
    sg=stream_gate[['stream','stream_gate_rank','stream_gate_final_score']].drop_duplicates('stream')
    q=q.drop(columns=[c for c in ['stream_gate_rank','stream_gate_final_score'] if c in q.columns], errors='ignore').merge(sg,on='stream',how='left')
    return q

def build_step2_candidate_base(full, stream_gate, use_top=50, step2_scope='watched8_all_members'):
    """Build the broad Step 2 evidence table BEFORE manual Step 3 deletion.

    This is intentionally NOT the old Q2 top2-core/top1-member choke.
    Step 2 is the bucket/evidence lab: rows are scored and bucketed so the
    user can decide in Step 3 what to keep/delete.

    step2_scope options:
    - watched8_all_members: default. N47+3 streams x all watched-8 cores x all 3 members.
    - watched8_positive_support: watched-8 rows with at least one profile support signal.
    - full120_all_members: N47+3 streams x all 120 cores x all 360 members. Large but available for audit.
    - legacy_q2_balanced: old repaired Q2 comparison only, not default.
    """
    gate_col=f'in_stream_gate_top{use_top}'
    if gate_col not in stream_gate.columns:
        stream_gate=stream_gate.copy()
        stream_gate[gate_col]=pd.to_numeric(stream_gate.get('stream_gate_rank'), errors='coerce').le(use_top)
    streams=set(stream_gate.loc[stream_gate[gate_col], 'stream'])
    w=full[full['stream'].isin(streams)].copy()
    scope=str(step2_scope or 'watched8_all_members')
    if scope == 'watched8_all_members':
        base=w[w['is_watched8_core']].copy()
    elif scope == 'watched8_positive_support':
        base=w[(w['is_watched8_core']) & (pd.to_numeric(w.get('all_support_count',0), errors='coerce').fillna(0).gt(0))].copy()
    elif scope == 'full120_all_members':
        base=w.copy()
    elif scope == 'legacy_q2_balanced':
        return build_q2_contenders(full, stream_gate, use_top=use_top, q2_mode='balanced_repair')
    else:
        raise ValueError(f'Unknown step2_scope: {step2_scope}')
    sg=stream_gate[['stream','stream_gate_rank','stream_gate_final_score']].drop_duplicates('stream')
    base=base.drop(columns=[c for c in ['stream_gate_rank','stream_gate_final_score'] if c in base.columns], errors='ignore').merge(sg,on='stream',how='left')
    base['step2_scope']=scope
    # keep old q2 field names available for compatibility, but they no longer mean an early choke
    base['q2_mode']=scope
    base['q2_core_rank_in_stream']=base.get('core_rank_in_stream')
    base['q2_member_rank_in_stream_core']=base.get('member_rank_in_stream_core')
    return base

# ---------- Step2 transition overlay ----------
def transition_tables(hist, through):
    """Build corrected Step 2 transition lift tables from history.

    This follows the corrected Step 2 concept: score seed->member compatibility
    on existing contender rows. It uses lift-style points rather than raw hit-rate
    scale so x15 does not swamp the profile score.
    """
    h=hist[hist['draw_date'].le(through)].sort_values(['stream','draw_date']).copy()
    h['prev_base4']=h.groupby('stream')['base4'].shift(1)
    h['prev_parity']=h['prev_base4'].map(parity_pattern)
    h['prev_parity_count']=h['prev_base4'].map(parity_count)
    h['prev_sum_bucket']=h['prev_base4'].map(lambda x:sum_bucket(digsum(x)))
    h=h[(h['core'].isin(WATCHED8)) & h['prev_base4'].notna() & h['member'].ne('')].copy()
    h['member_parity']=h['member'].map(parity_pattern)
    h['member_parity_count']=h['member'].map(parity_count)
    h['member_sum_bucket']=h['member'].map(lambda x:sum_bucket(digsum(x)))
    h['core_family']=h['core'].map(lambda c: 'EVEN_HEAVY_2E1O' if str(c).zfill(3) in {'027','067'} else 'ODD_HEAVY_1E2O')
    def lift_tab(src, tgt):
        if h.empty: return pd.DataFrame()
        # conditional hits for src->target
        t=h.groupby([src,tgt], as_index=False).size().rename(columns={'size':'hit_count'})
        rowtot=h.groupby(src, as_index=False).size().rename(columns={'size':'row_total'})
        colt=h.groupby(tgt, as_index=False).size().rename(columns={'size':'target_total'})
        total=len(h)
        t=t.merge(rowtot,on=src,how='left').merge(colt,on=tgt,how='left')
        t['row_pct']=t['hit_count']/t['row_total']
        t['base_pct']=t['target_total']/total
        t['lift']=t['row_pct']/t['base_pct'].replace(0,np.nan)
        t['lift']=t['lift'].replace([np.inf,-np.inf],np.nan).fillna(0.0)
        return t
    return {
        'sum_member_sum': lift_tab('prev_sum_bucket','member_sum_bucket'),
        'seedpat_memberpc': lift_tab('prev_parity','member_parity_count'),
        'seedpat_memberpat': lift_tab('prev_parity','member_parity'),
        'seedpat_corefam': lift_tab('prev_parity','core_family'),
        'seedpc_memberpc': lift_tab('prev_parity_count','member_parity_count'),
    }

def _lift_point(lift, weight):
    try:
        lift=float(lift)
    except Exception:
        return 0.0
    if lift<=0 or not np.isfinite(lift): return 0.0
    return math.log(lift)*weight

def lookup_transition_score(row, tabs):
    score=0.0; good=0; bad=0; details=[]
    core=str(row.get('core')).zfill(3)
    member=str(row.get('member', row.get('boxed_member',''))).zfill(4)
    seed_pat=row.get('seed_parity') or row.get('seed_parity_pattern')
    seed_pc=row.get('seed_parity_count')
    seed_sum=row.get('seed_sum_bucket') or sum_bucket(digsum(row.get('seed')))
    mem_pat=parity_pattern(member)
    mem_pc=parity_count(member)
    mem_sum=sum_bucket(digsum(member))
    corefam='EVEN_HEAVY_2E1O' if core in {'027','067'} else 'ODD_HEAVY_1E2O'
    checks=[
        ('sum_member_sum','prev_sum_bucket','member_sum_bucket',seed_sum,mem_sum,5.0),
        ('seedpat_memberpc','prev_parity','member_parity_count',seed_pat,mem_pc,6.0),
        ('seedpat_memberpat','prev_parity','member_parity',seed_pat,mem_pat,2.0),
        ('seedpat_corefam','prev_parity','core_family',seed_pat,corefam,5.0),
        ('seedpc_memberpc','prev_parity_count','member_parity_count',seed_pc,mem_pc,2.0),
    ]
    result={}
    for key,src_col,tgt_col,src_val,tgt_val,weight in checks:
        t=tabs.get(key,pd.DataFrame())
        if t.empty:
            bad+=1; details.append(f'{key}:missing_table'); continue
        m=t[(t[src_col].astype(str).eq(str(src_val))) & (t[tgt_col].astype(str).eq(str(tgt_val)))]
        if m.empty:
            bad+=1; details.append(f'{key}:no_hist'); continue
        rr=m.iloc[0]
        lift=float(rr['lift']); pts=_lift_point(lift, weight)
        score += pts
        if pts>0: good += 1
        elif pts<0: bad += 1
        details.append(f"{key}:lift={lift:.3f};pts={pts:.3f};hits={int(rr['hit_count'])}/row={int(rr['row_total'])}")
        result[f'{key}_lift']=lift
        result[f'{key}_points']=pts
        result[f'{key}_hits']=int(rr['hit_count'])
        result[f'{key}_row_total']=int(rr['row_total'])
    result.update({'transition_compat_score':score,'good_transition_count':good,'bad_transition_count':bad,'transition_detail':'; '.join(details)})
    return pd.Series(result)

def apply_step2_transition(q2, hist, through, x=15):
    tabs=transition_tables(hist, through)
    out=q2.copy()
    if out.empty:
        out['transition_compat_score']=[]; return out
    trans=out.apply(lambda r: lookup_transition_score(r, tabs), axis=1)
    out=pd.concat([out.reset_index(drop=True), trans.reset_index(drop=True)], axis=1)
    out['final_plus_transition_x15']=pd.to_numeric(out['profile_final_member_score'],errors='coerce').fillna(0) + x*pd.to_numeric(out['transition_compat_score'],errors='coerce').fillna(0)
    out=out.sort_values(['play_date','final_plus_transition_x15','profile_final_member_score','stream_gate_rank'], ascending=[True,False,False,True]).copy()
    out['step2_x15_rank']=out.groupby('play_date').cumcount()+1
    out['step2_no_hard_delete']=True
    return out

# ---------- Step3 formulas ----------
def add_rowcount_buckets(df, basis_df=None):
    """Attach core/stream row-count bucket labels.

    Important V17 correction:
    - The rows displayed in Step 2 can be the full broad member table.
    - The bucket counts may be calculated from a selected *evidence basis*
      such as final_x15>0, major>=3, major>=4, good>=1/no_bad, etc.
    This prevents the meaningless Cartesian count problem where every watched
    core shows 150 simply because 50 streams × 3 members were enumerated.
    """
    d=df.copy()
    b=d.copy() if basis_df is None else basis_df.copy()
    bucket_prefixes=(
        'core_row_count','mean_core_row_count','median_core_row_count','core_is_',
        'core_above_','core_below_','core_at_','stream_row_count','mean_stream_row_count',
        'median_stream_row_count','stream_is_','stream_above_','stream_below_','stream_at_'
    )
    drop_cols=[c for c in d.columns if c.startswith(bucket_prefixes) or c in ['core_row_count_rank','stream_row_count_rank']]
    if drop_cols:
        d=d.drop(columns=drop_cols)

    core_cols=['play_date','core','core_row_count','mean_core_row_count','median_core_row_count',
               'core_is_max','core_is_min',
               'core_above_mean','core_at_mean','core_below_mean','core_at_or_above_mean','core_at_or_below_mean',
               'core_above_median','core_at_median','core_below_median','core_at_or_above_median','core_at_or_below_median',
               'core_row_count_rank']
    stream_cols=['play_date','stream','stream_row_count','mean_stream_row_count','median_stream_row_count',
                 'stream_is_max','stream_is_min',
                 'stream_above_mean','stream_at_mean','stream_below_mean','stream_at_or_above_mean','stream_at_or_below_mean',
                 'stream_above_median','stream_at_median','stream_below_median','stream_at_or_above_median','stream_at_or_below_median',
                 'stream_row_count_rank']

    if d.empty:
        for col in core_cols + stream_cols:
            if col not in d.columns and col not in ['play_date','core','stream']:
                d[col] = pd.Series(dtype='object')
        return d, pd.DataFrame(columns=core_cols), pd.DataFrame(columns=stream_cols)

    required=['play_date','core','stream']
    missing=[c for c in required if c not in d.columns]
    if missing:
        raise KeyError(f"Cannot build row-count buckets; missing required columns: {missing}")
    missing_b=[c for c in required if c not in b.columns]
    if missing_b:
        raise KeyError(f"Cannot build row-count buckets from selected evidence basis; missing columns: {missing_b}")

    # Universe for bucket comparisons comes from displayed rows so zero-count cores/streams stay visible.
    all_cores=d[['play_date','core']].drop_duplicates().copy()
    all_streams=d[['play_date','stream']].drop_duplicates().copy()
    basis_core=b.groupby(['play_date','core'], as_index=False).size().rename(columns={'size':'core_row_count'}) if not b.empty else pd.DataFrame(columns=['play_date','core','core_row_count'])
    basis_stream=b.groupby(['play_date','stream'], as_index=False).size().rename(columns={'size':'stream_row_count'}) if not b.empty else pd.DataFrame(columns=['play_date','stream','stream_row_count'])
    core_counts=all_cores.merge(basis_core,on=['play_date','core'],how='left')
    stream_counts=all_streams.merge(basis_stream,on=['play_date','stream'],how='left')
    core_counts['core_row_count']=pd.to_numeric(core_counts['core_row_count'], errors='coerce').fillna(0).astype(int)
    stream_counts['stream_row_count']=pd.to_numeric(stream_counts['stream_row_count'], errors='coerce').fillna(0).astype(int)

    cc=[]; ss=[]
    for date,g in core_counts.groupby('play_date'):
        mean=float(g.core_row_count.mean()); med=float(g.core_row_count.median()); maxv=g.core_row_count.max(); minv=g.core_row_count.min()
        gg=g.copy(); gg['mean_core_row_count']=mean; gg['median_core_row_count']=med
        gg['core_is_max']=gg.core_row_count.eq(maxv); gg['core_is_min']=gg.core_row_count.eq(minv)
        gg['core_above_mean']=gg.core_row_count.gt(mean); gg['core_at_mean']=gg.core_row_count.eq(mean); gg['core_below_mean']=gg.core_row_count.lt(mean)
        gg['core_at_or_above_mean']=gg.core_row_count.ge(mean); gg['core_at_or_below_mean']=gg.core_row_count.le(mean)
        gg['core_above_median']=gg.core_row_count.gt(med); gg['core_at_median']=gg.core_row_count.eq(med); gg['core_below_median']=gg.core_row_count.lt(med)
        gg['core_at_or_above_median']=gg.core_row_count.ge(med); gg['core_at_or_below_median']=gg.core_row_count.le(med)
        gg=gg.sort_values(['core_row_count','core'], ascending=[False,True]); gg['core_row_count_rank']=range(1,len(gg)+1)
        cc.append(gg)
    for date,g in stream_counts.groupby('play_date'):
        mean=float(g.stream_row_count.mean()); med=float(g.stream_row_count.median()); maxv=g.stream_row_count.max(); minv=g.stream_row_count.min()
        gg=g.copy(); gg['mean_stream_row_count']=mean; gg['median_stream_row_count']=med
        gg['stream_is_max']=gg.stream_row_count.eq(maxv); gg['stream_is_min']=gg.stream_row_count.eq(minv)
        gg['stream_above_mean']=gg.stream_row_count.gt(mean); gg['stream_at_mean']=gg.stream_row_count.eq(mean); gg['stream_below_mean']=gg.stream_row_count.lt(mean)
        gg['stream_at_or_above_mean']=gg.stream_row_count.ge(mean); gg['stream_at_or_below_mean']=gg.stream_row_count.le(mean)
        gg['stream_above_median']=gg.stream_row_count.gt(med); gg['stream_at_median']=gg.stream_row_count.eq(med); gg['stream_below_median']=gg.stream_row_count.lt(med)
        gg['stream_at_or_above_median']=gg.stream_row_count.ge(med); gg['stream_at_or_below_median']=gg.stream_row_count.le(med)
        gg=gg.sort_values(['stream_row_count','stream'], ascending=[False,True]); gg['stream_row_count_rank']=range(1,len(gg)+1)
        ss.append(gg)
    core_counts=pd.concat(cc, ignore_index=True) if cc else pd.DataFrame(columns=core_cols)
    stream_counts=pd.concat(ss, ignore_index=True) if ss else pd.DataFrame(columns=stream_cols)
    d=d.merge(core_counts,on=['play_date','core'],how='left').merge(stream_counts,on=['play_date','stream'],how='left')
    return d, core_counts, stream_counts

def step3_formula_grid(step2, winner_targets=None):
    d, core_counts, stream_counts=add_rowcount_buckets(step2)
    qual={
      'all_rows': lambda x: pd.Series(True,index=x.index),
      'safe_no_bad': lambda x: x['bad_transition_count'].fillna(0).le(0),
      'aggressive_good1_no_bad': lambda x: x['good_transition_count'].fillna(0).ge(1) & x['bad_transition_count'].fillna(0).le(0),
      'major_ge4': lambda x: x['major_support_count'].fillna(0).ge(4),
      'major_ge4_and_aggressive': lambda x: x['major_support_count'].fillna(0).ge(4) & x['good_transition_count'].fillna(0).ge(1) & x['bad_transition_count'].fillna(0).le(0),
    }
    core_methods={
      'CORE_all': lambda x: pd.Series(True,index=x.index),
      'CORE_max': lambda x: x['core_is_max'],
      'CORE_top2': lambda x: x['core_row_count_rank'].le(2),
      'CORE_top3': lambda x: x['core_row_count_rank'].le(3),
      'CORE_above_median': lambda x: x['core_above_median'],
      'CORE_at_or_above_median': lambda x: x['core_at_or_above_median'],
      'CORE_minmax_rescue': lambda x: x['core_is_max'] | x['core_is_min'],
      'CORE_top2_minmax_rescue': lambda x: x['core_row_count_rank'].le(2) | x['core_is_min'] | x['core_is_max'],
    }
    stream_methods={
      'STREAM_all': lambda x: pd.Series(True,index=x.index),
      'STREAM_above_median': lambda x: x['stream_above_median'],
      'STREAM_at_or_above_median': lambda x: x['stream_at_or_above_median'],
      'STREAM_at_or_below_median': lambda x: x['stream_at_or_below_median'],
      'STREAM_remove_highest5_crowded': lambda x: ~x['stream_row_count_rank'].le(5),
      'STREAM_remove_highest10_crowded': lambda x: ~x['stream_row_count_rank'].le(10),
    }
    summaries=[]; details=[]; playlists={}
    for qn,qf in qual.items():
      qd=d[qf(d)].copy()
      if qd.empty: continue
      qd,_,_=add_rowcount_buckets(qd)
      for cn,cf in core_methods.items():
        for sn,sf in stream_methods.items():
          keep=qd[cf(qd) & sf(qd)].copy()
          keep=keep.sort_values(['play_date','final_plus_transition_x15','profile_final_member_score','stream_gate_rank'], ascending=[True,False,False,True]).copy()
          keep['step3_rank']=keep.groupby('play_date').cumcount()+1
          variant=f'{qn}__{cn}__{sn}'
          if len(keep): playlists[variant]=keep
          row={'variant':variant,'qualification':qn,'core_method':cn,'stream_method':sn,'rows_total':len(keep)}
          for topn in [47,50]:
            row[f'rows_top{topn}']=int(min(len(keep), topn))
          # winner audit
          if winner_targets is not None and not winner_targets.empty:
            wt=winner_targets.copy()
            wt['core']=wt['core'].astype(str).str.zfill(3); wt['member']=wt['member'].astype(str).str.zfill(4)
            caps=[]
            for _,w in wt.iterrows():
              m=keep[(keep['play_date'].astype(str).eq(str(w['play_date']))) & (keep['stream'].astype(str).eq(str(w['stream']))) & (keep['core'].astype(str).str.zfill(3).eq(w['core'])) & (keep['member'].astype(str).str.zfill(4).eq(w['member']))]
              present=not m.empty
              rank=int(m.iloc[0]['step3_rank']) if present else None
              caps.append({'play_date':w['play_date'],'stream':w['stream'],'core':w['core'],'member':w['member'],'variant':variant,'present_after_step3':present,'rank_after_step3':rank,'top47_capture':present and rank<=47,'top50_capture':present and rank<=50,'blind_plays_if_play_all_variant_rows':len(keep[keep['play_date'].astype(str).eq(str(w['play_date']))])})
            dd=pd.DataFrame(caps); details.append(dd)
            row['winner_rows_present']=int(dd['present_after_step3'].sum())
            row['winner_top47_capture']=int(dd['top47_capture'].sum())
            row['winner_top50_capture']=int(dd['top50_capture'].sum())
            row['winner_targets']=len(dd)
          summaries.append(row)
    summary=pd.DataFrame(summaries)
    detail=pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    return d, core_counts, stream_counts, summary, detail, playlists

# ---------- winner targets from history ----------
def winner_targets_from_history(hist, play_date):
    h=hist[(hist['draw_date'].eq(pd.to_datetime(play_date).strftime('%Y-%m-%d'))) & (hist['core'].isin(WATCHED8))].copy()
    return h.rename(columns={'draw_date':'play_date','base4':'winner'})[['play_date','stream','core','member','winner']]

# ---------- run one date ----------
def run_date(history_path, profile_dir, out_dir, play_date, history_through=None, exclude_az_md=True, stream_gate_top=47, winner_targets_path=None):
    out=Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    hist0=read_history(history_path)
    hist, exclusion_audit=apply_exclusions(hist0, exclude_az_md=exclude_az_md)
    play_date=pd.to_datetime(play_date).strftime('%Y-%m-%d')
    if history_through is None:
        history_through=(pd.to_datetime(play_date)-pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    else: history_through=pd.to_datetime(history_through).strftime('%Y-%m-%d')
    prof=load_profiles(profile_dir)
    events=derive_seed_events(hist, history_through, play_date)
    full=build_full_enumeration(hist, events, prof)
    gate=build_stream_gate(full, hist=hist, history_through=history_through, mode='baseline_n47')
    q2=build_q2_contenders(full, gate, use_top=stream_gate_top)
    step2=apply_step2_transition(q2, hist, history_through, x=15)
    # winner targets: prefer provided list if available, else history if play_date exists
    if winner_targets_path and Path(winner_targets_path).exists():
        wt=pd.read_csv(winner_targets_path, dtype=str)
        wt=wt[wt['play_date'].astype(str).eq(play_date)].copy()
    else:
        wt=winner_targets_from_history(hist0, play_date)
    step3_annot, core_counts, stream_counts, grid, detail, playlists=step3_formula_grid(step2, wt)
    # write files
    exclusion_audit.to_csv(out/'00_STEP0_EXCLUSION_AUDIT.csv', index=False)
    pd.DataFrame([{'BUILD':BUILD,'history_rows_loaded':len(hist0),'history_rows_after_exclusion':len(hist),'history_through':history_through,'play_date':play_date,'streams':events['stream'].nunique(),'full_enumeration_rows':len(full),'q2_rows':len(q2),'step2_rows':len(step2),'winner_targets':len(wt),'mirror_filter_used':False}]).to_csv(out/'00_RUN_SUMMARY.csv', index=False)
    events.to_csv(out/'01_STEP0_SEEDS_FOR_PLAY_DATE.csv', index=False)
    full.to_csv(out/'02_STEP1_FULL120_FULL_ENUMERATION.csv', index=False)
    gate.to_csv(out/'03_STREAM_GATE_BASELINE_N47_WITH_WINNER_SCALE_AUDIT.csv', index=False)
    q2.to_csv(out/'04_Q2_BASE_CONTENDER_ROWS_CORETOP2_MEMTOP1.csv', index=False)
    step2.to_csv(out/'05_STEP2_CORRECTED_TRANSITION_X15_NO_HARD_DELETE.csv', index=False)
    step3_annot.to_csv(out/'06_STEP3_ROWS_WITH_CORE_STREAM_BUCKETS.csv', index=False)
    core_counts.to_csv(out/'07_STEP3_CORE_ROWCOUNT_MEAN_MEDIAN.csv', index=False)
    stream_counts.to_csv(out/'08_STEP3_STREAM_ROWCOUNT_MEAN_MEDIAN.csv', index=False)
    grid.to_csv(out/'09_STEP3_REDUCTION_FORMULA_GRID_SUMMARY.csv', index=False)
    detail.to_csv(out/'10_STEP3_WINNER_DETAIL_BY_FORMULA.csv', index=False)
    wt.to_csv(out/'11_WINNER_TARGETS_USED.csv', index=False)
    # write top few playlists by capture/rows
    if not grid.empty:
        g=grid.copy()
        for c in ['winner_top50_capture','winner_top47_capture','rows_total']:
            if c not in g: g[c]=0
        g['has_any_rows']=g['rows_total'].fillna(0).gt(0)
        g=g.sort_values(['winner_top50_capture','winner_top47_capture','has_any_rows','rows_total'], ascending=[False,False,False,True]).head(10)
        g.to_csv(out/'12_TOP_10_STEP3_FORMULAS.csv', index=False)
        for _,r in g.head(5).iterrows():
            var=r['variant']; pl=playlists.get(var)
            if pl is not None:
                safe=re.sub(r'[^A-Za-z0-9_]+','_',var)[:140]
                pl.to_csv(out/f'PLAYLIST_{safe}.csv', index=False)
    report=f"""{BUILD}\n\nPlay date: {play_date}\nHistory through: {history_through}\nStreams after Step 0: {events['stream'].nunique()}\nStep 1 full enumeration rows: {len(full)}\nActive stream gate rows selected Top{stream_gate_top}: {int(gate[f'in_stream_gate_top{stream_gate_top}'].sum())}\nQ2 base contender rows: {len(q2)}\nStep 2 x15 rows: {len(step2)}\nWinner targets available for audit: {len(wt)}\n\nLocked behavior:\n- Mirror hard filtering is NOT used.\n- Step 2 is overlay only; no hard deletion.\n- Step 3 tests formulas; it does not lock final reduction.\n- Active stream gate default is baseline N47; winner-scale/cadence is exported as audit.
- Top47 and Top50 rescue capture are both reported.\n"""
    (out/'00_PLAIN_LANGUAGE_REPORT.txt').write_text(report, encoding='utf-8')
    # zip date outputs
    zip_path=out.parent/f'{out.name}.zip'
    with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file(): z.write(p, p.relative_to(out))
    return zip_path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--history', required=True)
    ap.add_argument('--profiles', default='profiles')
    ap.add_argument('--out', default='OUT_DAILY_LADDER')
    ap.add_argument('--play-date', required=True)
    ap.add_argument('--history-through', default=None)
    ap.add_argument('--include-az-md', action='store_true')
    ap.add_argument('--stream-gate-top', type=int, default=47)
    ap.add_argument('--winner-targets', default=None)
    args=ap.parse_args()
    zp=run_date(args.history,args.profiles,args.out,args.play_date,args.history_through,exclude_az_md=not args.include_az_md,stream_gate_top=args.stream_gate_top,winner_targets_path=args.winner_targets)
    print(f'Wrote {zp}')
# original CLI disabled; V4 CLI below


# ---------- V4 exact small-list final playlist ----------
def _norm_core_col(s):
    return s.astype(str).str.replace(r'\.0$','',regex=True).str.zfill(3)

def _norm_member_col(s):
    return s.astype(str).str.replace(r'\.0$','',regex=True).str.zfill(4)

def build_exact_smalllist_playlist(step2_df, main_limit=47, borderline_count=3):
    """Build the exact small-list architecture that caught Oregon 06/19.

    Required standard:
    - major_support_count >= 4
    - bad_transition_count == 0
    - good_transition_count >= 1
    - choose the core(s) with maximum qualified row count
    - rank by final_plus_transition_x15

    Borderline rows are NOT a separate fishing pool. They are the next rows that
    met the same support/transition standard but missed the final kept core rule.
    If the main qualified list exceeds main_limit, rows main_limit+1 through
    main_limit+borderline_count are the borderline rows by the same ranking.
    """
    df=step2_df.copy()
    # normalize important columns
    if 'boxed_member' in df.columns and 'member' not in df.columns:
        df['member']=df['boxed_member']
    if 'member' in df.columns:
        df['member']=_norm_member_col(df['member'])
    elif 'boxed_member' in df.columns:
        df['member']=_norm_member_col(df['boxed_member'])
    if 'boxed_member' not in df.columns:
        df['boxed_member']=df.get('member','')
    if 'core' in df.columns:
        df['core']=_norm_core_col(df['core'])
    needed=['major_support_count','bad_transition_count','good_transition_count','final_plus_transition_x15','profile_final_member_score','transition_compat_score']
    for c in needed:
        if c not in df.columns:
            df[c]=0
        df[c]=pd.to_numeric(df[c], errors='coerce').fillna(0)
    if 'stream_gate_rank' not in df.columns: df['stream_gate_rank']=999
    if 'seed' not in df.columns: df['seed']=''
    # V4 standard columns are explicit so the final playlist can be audited.
    # For original corrected Step2 files, these equal the saved support/transition behavior.
    # For history-built files, they prevent tiny/missing transition penalties from falsely
    # blocking the exact small-list architecture. This does not add a wide rescue pool.
    support_cols=['score_stream_core_usable','score_seed_trait_usable','score_stream_seed_trait_usable','score_cadence','score_member_role','score_exact_stream_core_member','score_c120_core_matrix','score_c120_member_matrix','score_c120_replacement_matrix','score_signal_count_summary']
    for c in support_cols:
        if c not in df.columns: df[c]=0
        df[c]=pd.to_numeric(df[c], errors='coerce').fillna(0)
    computed_support=(df[support_cols]>0).sum(axis=1)
    has_original_rich_step2 = ('score_files_fired' in df.columns and df['score_files_fired'].notna().any()) or ('branch_name' in df.columns)
    if has_original_rich_step2:
        # Preserve the exact saved Step2 architecture when the rich corrected Step2 file is supplied.
        df['v4_standard_major_support_count']=df['major_support_count']
        df['v4_standard_bad_transition_count']=df['bad_transition_count']
        df['v4_standard_good_transition_count']=df['good_transition_count']
    else:
        # History-built Step2 files do not contain every original rich score/support field.
        # Use the same standard concept, but compute support from available score columns
        # and ignore tiny non-negative transition noise as a hard bad flag.
        df['v4_standard_major_support_count']=pd.concat([df['major_support_count'], computed_support],axis=1).max(axis=1)
        df['v4_standard_bad_transition_count']=df['bad_transition_count'].where(df['transition_compat_score']<0,0)
        df['v4_standard_good_transition_count']=df['good_transition_count']
    qualified=df[(df['v4_standard_major_support_count']>=4)&(df['v4_standard_bad_transition_count']<=0)&(df['v4_standard_good_transition_count']>=1)].copy()
    if qualified.empty:
        cols=['playlist_rank','playlist_section','stream','seed','core','member','final_plus_transition_x15','profile_final_member_score','transition_compat_score','major_support_count','all_support_count','good_transition_count','bad_transition_count','score_files_fired','v4_standard_major_support_count','v4_standard_good_transition_count','v4_standard_bad_transition_count']
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=cols), qualified, pd.DataFrame()
    core_counts=qualified.groupby('core', as_index=False).size().rename(columns={'size':'qualified_core_row_count'})
    core_counts=core_counts.sort_values(['qualified_core_row_count','core'], ascending=[False,True]).copy()
    max_count=int(core_counts['qualified_core_row_count'].max())
    max_cores=core_counts.loc[core_counts['qualified_core_row_count'].eq(max_count),'core'].tolist()
    qualified=qualified.merge(core_counts,on='core',how='left')
    main_pool=qualified[qualified['core'].isin(max_cores)].copy()
    other_pool=qualified[~qualified['core'].isin(max_cores)].copy()
    sort_cols=['final_plus_transition_x15','profile_final_member_score','major_support_count','all_support_count','stream_gate_rank','stream']
    asc=[False,False,False,False,True,True]
    main_sorted=main_pool.sort_values(sort_cols, ascending=asc).reset_index(drop=True)
    if len(main_sorted)>main_limit:
        main=main_sorted.head(main_limit).copy()
        borderline=main_sorted.iloc[main_limit:main_limit+borderline_count].copy()
    else:
        main=main_sorted.copy()
        # borderline are qualified rows that met the same support/transition standard but just missed core max.
        borderline=other_pool.sort_values(['qualified_core_row_count']+sort_cols, ascending=[False]+asc).head(borderline_count).copy()
    main['playlist_rank']=range(1,len(main)+1)
    main['playlist_section']='MAIN_RECOMMENDED'
    if not borderline.empty:
        borderline=borderline.copy().reset_index(drop=True)
        borderline['playlist_rank']=range(len(main)+1, len(main)+1+len(borderline))
        borderline['playlist_section']='BORDERLINE_NEXT_3'
    cols=['playlist_rank','playlist_section','stream','seed','core','member','final_plus_transition_x15','profile_final_member_score','transition_compat_score','major_support_count','all_support_count','good_transition_count','bad_transition_count','score_files_fired','v4_standard_major_support_count','v4_standard_good_transition_count','v4_standard_bad_transition_count']
    for c in cols:
        if c not in main.columns: main[c]=''
        if c not in borderline.columns: borderline[c]=''
    return main[cols].copy(), borderline[cols].copy(), qualified, core_counts

def build_v4_outputs_from_step2_csv(step2_csv, out_dir, main_limit=47, borderline_count=3, winner_targets_path=None):
    out=Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    step2=pd.read_csv(step2_csv, dtype=str)
    main,borderline,qualified,core_counts=build_exact_smalllist_playlist(step2, main_limit=main_limit, borderline_count=borderline_count)
    final_with_borderline=pd.concat([main,borderline], ignore_index=True)
    main.to_csv(out/'13_FINAL_PLAYLIST_MAIN_RECOMMENDED.csv', index=False)
    borderline.to_csv(out/'14_BORDERLINE_NEXT_3.csv', index=False)
    final_with_borderline.to_csv(out/'15_FINAL_PLAYLIST_MAIN_PLUS_BORDERLINE.csv', index=False)
    qualified.to_csv(out/'16_QUALIFIED_ROWS_BEFORE_CORE_MAX_SELECTION.csv', index=False)
    core_counts.to_csv(out/'17_QUALIFIED_CORE_ROW_COUNTS.csv', index=False)
    # printable txt
    lines=[]
    lines.append(f"V4 Exact Small-List Playlist")
    lines.append(f"Main rows: {len(main)}")
    lines.append(f"Borderline rows: {len(borderline)}")
    lines.append(f"Total displayed rows: {len(final_with_borderline)}")
    lines.append("")
    for _,r in final_with_borderline.iterrows():
        label='MAIN' if r['playlist_section']=='MAIN_RECOMMENDED' else 'BORDERLINE'
        lines.append(f"{int(r['playlist_rank']):02d}. [{label}] {r['stream']} | MBR {r['member']} | Core {r['core']} | Seed {r['seed']} | Score {float(r['final_plus_transition_x15']):.3f}")
    (out/'15_FINAL_PLAYLIST_PRINTABLE.txt').write_text('\n'.join(lines), encoding='utf-8')
    # winner audit if supplied
    if winner_targets_path and Path(winner_targets_path).exists():
        wt=pd.read_csv(winner_targets_path, dtype=str)
        aud=[]
        f=final_with_borderline.copy()
        if 'member' in f.columns: f['member']=_norm_member_col(f['member'])
        if 'core' in f.columns: f['core']=_norm_core_col(f['core'])
        for _,w in wt.iterrows():
            wc=str(w.get('core','')).zfill(3)
            wm=str(w.get('member','')).zfill(4)
            ws=str(w.get('stream',''))
            m=f[(f['stream'].astype(str).eq(ws))&(f['core'].astype(str).eq(wc))&(f['member'].astype(str).eq(wm))]
            aud.append({'stream':ws,'core':wc,'member':wm,'present_in_final_or_borderline':not m.empty,'rank':int(m.iloc[0]['playlist_rank']) if not m.empty else None,'section':m.iloc[0]['playlist_section'] if not m.empty else ''})
        pd.DataFrame(aud).to_csv(out/'18_WINNER_AUDIT_IF_PROVIDED.csv', index=False)
    pd.DataFrame([{'BUILD':BUILD,'input_step2_csv':str(step2_csv),'main_rows':len(main),'borderline_rows':len(borderline),'total_main_plus_borderline':len(final_with_borderline),'main_limit':main_limit,'borderline_count':borderline_count,'mirror_filter_used':False,'step2_hard_delete_used':False,'architecture':'major_ge4_and_aggressive + CORE_max + sort final_plus_transition_x15; borderline next 3 same qualification'}]).to_csv(out/'00_V4_RUN_SUMMARY.csv', index=False)
    return final_with_borderline

def run_date_v4(history_path, profile_dir, out_dir, play_date, history_through=None, exclude_az_md=True, stream_gate_top=47, main_limit=47, borderline_count=3, winner_targets_path=None):
    """Run daily file builder, then V4 exact small-list final playlist."""
    out=Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    # run the inherited daily file generation first
    run_date(history_path, profile_dir, out, play_date, history_through, exclude_az_md, stream_gate_top, winner_targets_path)
    step2=out/'05_STEP2_CORRECTED_TRANSITION_X15_NO_HARD_DELETE.csv'
    final=build_v4_outputs_from_step2_csv(step2, out, main_limit=main_limit, borderline_count=borderline_count, winner_targets_path=winner_targets_path)
    # append V4 note
    note=f"""{BUILD}\n\nThis app builds the daily testing file the project kept needing:\n05_STEP2_CORRECTED_TRANSITION_X15_NO_HARD_DELETE.csv\n\nThen it applies the exact small-list architecture that caught Oregon 06/19:\n- major_support_count >= 4\n- bad_transition_count == 0\n- good_transition_count >= 1\n- keep core(s) with maximum qualified row count\n- sort by final_plus_transition_x15\n- output main list up to {main_limit}\n- output only the next {borderline_count} borderline rows that toed the same standard\n\nNo mirror hard filtering. No wide-pool rescue fishing. No attempt to save every other winner in this version.\n"""
    (out/'00_V4_PLAIN_LANGUAGE_REPORT.txt').write_text(note, encoding='utf-8')
    zip_path=out.parent/f'{out.name}.zip'
    with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file(): z.write(p,p.relative_to(out))
    return zip_path



# ---------- V5 manual core/stream reduction and N47+3 stream borderlines ----------
QUALIFICATION_METHODS = ['major_ge4_and_aggressive','major_ge4','aggressive_good1_no_bad','safe_no_bad','all_rows']
CORE_METHODS = ['CORE_max','CORE_all','CORE_top2','CORE_top3','CORE_above_median','CORE_at_or_above_median','CORE_minmax_rescue','CORE_top2_minmax_rescue']
STREAM_METHODS = ['STREAM_all','STREAM_above_median','STREAM_at_or_above_median','STREAM_at_or_below_median','STREAM_remove_highest5_crowded','STREAM_remove_highest10_crowded']

def read_seed_list(path):
    """Read optional current seed list TXT/CSV. Expected: StreamKey/stream + seed/result; date optional.
    This lets the app use a current 80-stream seed page when the history file is not fully current.
    """
    df=read_table_any(path)
    cols={c.lower().strip():c for c in df.columns}
    stream_col=None; seed_col=None; date_col=None
    for c in ['streamkey','stream','stream_name','streamname']:
        if c in cols: stream_col=cols[c]; break
    if stream_col is None and 'state' in cols and 'game' in cols:
        df['__stream']=df[cols['state']].astype(str).str.strip()+' | '+df[cols['game']].astype(str).str.strip()
        stream_col='__stream'
    for c in ['seed','result4','base4','result','winning_number']:
        if c in cols: seed_col=cols[c]; break
    for c in ['date','draw_date','seed_date','history_through']:
        if c in cols: date_col=cols[c]; break
    if stream_col is None or seed_col is None:
        raise ValueError('Seed list must include StreamKey/Stream or State+Game, plus Seed/Result4/Result.')
    out=pd.DataFrame({
        'stream':df[stream_col].astype(str).str.strip(),
        'seed_override':df[seed_col].map(norm4)
    })
    if date_col:
        out['seed_list_date']=pd.to_datetime(df[date_col], errors='coerce').dt.strftime('%Y-%m-%d')
    else:
        out['seed_list_date']=''
    out=out[out['stream'].ne('') & out['seed_override'].ne('')].drop_duplicates('stream', keep='last')
    return out

def apply_seed_override(events, seed_list):
    ev=events.copy()
    ev['seed_source']='history_latest_same_stream'
    ev['seed_list_date']=''
    if seed_list is None or seed_list.empty:
        return ev
    ev=ev.merge(seed_list, on='stream', how='left')
    m=ev['seed_override'].notna() & ev['seed_override'].astype(str).ne('')
    ev.loc[m,'seed']=ev.loc[m,'seed_override'].map(norm4)
    ev.loc[m,'seed_source']='uploaded_seed_list_override'
    ev.loc[m,'seed_list_date']=ev.loc[m,'seed_list_date'].fillna('')
    # recompute seed traits after override
    ev['seed_member']=ev['seed'].map(boxed_member)
    ev['seed_core']=ev['seed'].map(core_from_result)
    ev['seed_sum']=ev['seed'].map(digsum)
    ev['seed_sum_bucket']=ev['seed_sum'].map(sum_bucket)
    ev['seed_spread']=ev['seed'].map(spread)
    ev['seed_parity']=ev['seed'].map(parity_pattern)
    ev['seed_parity_count']=ev['seed'].map(parity_count)
    ev['seed_highlow']=ev['seed'].map(highlow)
    ev['seed_structure']=ev['seed'].map(structure4)
    ev=ev.drop(columns=[c for c in ['seed_override'] if c in ev.columns])
    return ev

def _ensure_numeric(df, cols):
    for c in cols:
        if c not in df.columns: df[c]=0
        df[c]=pd.to_numeric(df[c], errors='coerce').fillna(0)
    return df

def _standardize_step2_for_v5(step2):
    df=step2.copy()
    if 'boxed_member' in df.columns and 'member' not in df.columns: df['member']=df['boxed_member']
    if 'member' in df.columns: df['member']=_norm_member_col(df['member'])
    if 'core' in df.columns: df['core']=_norm_core_col(df['core'])
    for c in ['seed','member','core','stream']:
        if c not in df.columns: df[c]=''
    df['seed']=df['seed'].map(norm4)
    _ensure_numeric(df, ['major_support_count','all_support_count','good_transition_count','bad_transition_count','transition_compat_score','final_plus_transition_x15','profile_final_member_score','stream_gate_rank'])
    support_cols=['score_stream_core_usable','score_seed_trait_usable','score_stream_seed_trait_usable','score_cadence','score_member_role','score_exact_stream_core_member','score_c120_core_matrix','score_c120_member_matrix','score_c120_replacement_matrix','score_signal_count_summary']
    for c in support_cols:
        if c not in df.columns: df[c]=0
        df[c]=pd.to_numeric(df[c], errors='coerce').fillna(0)
    computed_support=(df[support_cols]>0).sum(axis=1)
    has_original_rich_step2 = ('score_files_fired' in df.columns and df['score_files_fired'].notna().any()) or ('branch_name' in df.columns)
    if has_original_rich_step2:
        df['v5_standard_major_support_count']=df['major_support_count']
        df['v5_standard_bad_transition_count']=df['bad_transition_count']
        df['v5_standard_good_transition_count']=df['good_transition_count']
    else:
        df['v5_standard_major_support_count']=pd.concat([df['major_support_count'], computed_support],axis=1).max(axis=1)
        df['v5_standard_bad_transition_count']=df['bad_transition_count'].where(df['transition_compat_score']<0,0)
        df['v5_standard_good_transition_count']=df['good_transition_count']
    if 'play_date' not in df.columns: df['play_date']=''
    if 'history_through' not in df.columns: df['history_through']=''
    if 'seed_list_date' not in df.columns: df['seed_list_date']=''
    return df

def build_v5_manual_playlist_from_step2(step2_df, out_dir, qualification='major_ge4_and_aggressive', core_method='CORE_max', stream_method='STREAM_all', main_stream_gate=47, stream_borderline_count=3, include_stream_borderlines_in_main=False, main_limit=47):
    out=Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df=_standardize_step2_for_v5(step2_df)
    # Apply qualification first.
    qual_masks={
        'all_rows': pd.Series(True,index=df.index),
        'safe_no_bad': df['v5_standard_bad_transition_count'].le(0),
        'aggressive_good1_no_bad': df['v5_standard_good_transition_count'].ge(1) & df['v5_standard_bad_transition_count'].le(0),
        'major_ge4': df['v5_standard_major_support_count'].ge(4),
        'major_ge4_and_aggressive': df['v5_standard_major_support_count'].ge(4) & df['v5_standard_good_transition_count'].ge(1) & df['v5_standard_bad_transition_count'].le(0),
    }
    if qualification not in qual_masks: raise ValueError(f'Unknown qualification {qualification}')
    q=df[qual_masks[qualification]].copy()
    if q.empty:
        q.to_csv(out/'13_FULL_RANKED_PLAYLIST_SELECTED_METHOD.csv', index=False)
        q.to_csv(out/'14_RECOMMENDED_PLAYLIST_MAIN.csv', index=False)
        q.to_csv(out/'15_STREAM_BORDERLINE_ROWS_N47_PLUS3.csv', index=False)
        pd.DataFrame([{'BUILD':BUILD,'rows':0,'qualification':qualification,'core_method':core_method,'stream_method':stream_method}]).to_csv(out/'00_V5_RUN_SUMMARY.csv', index=False)
        return q
    q, core_counts, stream_counts = add_rowcount_buckets(q)
    core_masks={
        'CORE_all': pd.Series(True,index=q.index),
        'CORE_max': q['core_is_max'],
        'CORE_top2': q['core_row_count_rank'].le(2),
        'CORE_top3': q['core_row_count_rank'].le(3),
        'CORE_above_median': q['core_above_median'],
        'CORE_at_or_above_median': q['core_at_or_above_median'],
        'CORE_minmax_rescue': q['core_is_max'] | q['core_is_min'],
        'CORE_top2_minmax_rescue': q['core_row_count_rank'].le(2) | q['core_is_min'] | q['core_is_max'],
    }
    stream_masks={
        'STREAM_all': pd.Series(True,index=q.index),
        'STREAM_above_median': q['stream_above_median'],
        'STREAM_at_or_above_median': q['stream_at_or_above_median'],
        'STREAM_at_or_below_median': q['stream_at_or_below_median'],
        'STREAM_remove_highest5_crowded': ~q['stream_row_count_rank'].le(5),
        'STREAM_remove_highest10_crowded': ~q['stream_row_count_rank'].le(10),
    }
    if core_method not in core_masks: raise ValueError(f'Unknown core_method {core_method}')
    if stream_method not in stream_masks: raise ValueError(f'Unknown stream_method {stream_method}')
    selected=q[core_masks[core_method] & stream_masks[stream_method]].copy()
    top_stream_limit=int(main_stream_gate)+int(stream_borderline_count)
    if not include_stream_borderlines_in_main:
        main_candidate=selected[selected['stream_gate_rank'].le(int(main_stream_gate))].copy()
        borderline_candidate=selected[(selected['stream_gate_rank'].gt(int(main_stream_gate))) & (selected['stream_gate_rank'].le(top_stream_limit))].copy()
    else:
        main_candidate=selected[selected['stream_gate_rank'].le(top_stream_limit)].copy()
        borderline_candidate=selected[(selected['stream_gate_rank'].gt(int(main_stream_gate))) & (selected['stream_gate_rank'].le(top_stream_limit))].copy()
    sort_cols=['final_plus_transition_x15','profile_final_member_score','v5_standard_major_support_count','all_support_count','stream_gate_rank','stream','core','member']
    asc=[False,False,False,False,True,True,True,True]
    full_ranked=selected.sort_values(sort_cols, ascending=asc).reset_index(drop=True)
    full_ranked['full_playlist_rank']=range(1,len(full_ranked)+1)
    rank_key_cols=['play_date','stream','core','member','seed']
    rank_map=full_ranked[rank_key_cols+['full_playlist_rank']].copy()
    main=main_candidate.sort_values(sort_cols, ascending=asc).head(int(main_limit)).reset_index(drop=True)
    main=main.merge(rank_map,on=rank_key_cols,how='left')
    main['playlist_rank']=range(1,len(main)+1)
    main['playlist_section']='MAIN_N47_RECOMMENDED'
    border=borderline_candidate.sort_values(sort_cols, ascending=asc).reset_index(drop=True)
    border=border.merge(rank_map,on=rank_key_cols,how='left')
    border['playlist_rank']=range(len(main)+1,len(main)+1+len(border))
    border['playlist_section']='STREAM_BORDERLINE_RANK_48_50'
    final=pd.concat([main,border], ignore_index=True)
    # enrich display columns
    for dfx in [full_ranked, main, border, final]:
        if dfx.empty: continue
        dfx['state_abbrev']=dfx['stream'].astype(str).str.split('|').str[0].str.strip()
        dfx['game_name']=dfx['stream'].astype(str).str.split('|').str[1:].str.join('|').str.strip()
        dfx['stream_gate_section']=np.where(dfx['stream_gate_rank'].le(int(main_stream_gate)),'N47_MAIN','N47_PLUS3_STREAM_BORDERLINE')
        dfx['reason_code']=f'{qualification} + {core_method} + {stream_method}'
    # ordered columns
    cols=['playlist_rank','playlist_section','full_playlist_rank','play_date','history_through','seed_list_date','stream_gate_section','stream_gate_rank','state_abbrev','game_name','stream','seed','core','member','final_plus_transition_x15','profile_final_member_score','transition_compat_score','v5_standard_major_support_count','v5_standard_good_transition_count','v5_standard_bad_transition_count','major_support_count','all_support_count','good_transition_count','bad_transition_count','core_row_count','core_row_count_rank','stream_row_count','stream_row_count_rank','reason_code','transition_detail']
    for dfx in [full_ranked, main, border, final]:
        for c in cols:
            if c not in dfx.columns: dfx[c]=''
    full_ranked.to_csv(out/'13_FULL_RANKED_PLAYLIST_SELECTED_METHOD.csv', index=False)
    main[cols].to_csv(out/'14_RECOMMENDED_PLAYLIST_MAIN.csv', index=False)
    border[cols].to_csv(out/'15_STREAM_BORDERLINE_ROWS_N47_PLUS3.csv', index=False)
    final[cols].to_csv(out/'16_MAIN_PLUS_STREAM_BORDERLINE_PLAYLIST.csv', index=False)
    core_counts.to_csv(out/'17_SELECTED_METHOD_CORE_MEAN_MEDIAN_COUNTS.csv', index=False)
    stream_counts.to_csv(out/'18_SELECTED_METHOD_STREAM_MEAN_MEDIAN_COUNTS.csv', index=False)
    # TXT output
    lines=[]
    pd0=final['play_date'].dropna().astype(str).iloc[0] if len(final) and 'play_date' in final else ''
    ht0=final['history_through'].dropna().astype(str).iloc[0] if len(final) and 'history_through' in final else ''
    sld=final['seed_list_date'].dropna().astype(str).iloc[0] if len(final) and 'seed_list_date' in final and final['seed_list_date'].astype(str).ne('').any() else ''
    lines.append(f'{BUILD}')
    lines.append(f'PLAYLIST_DATE: {pd0}')
    lines.append(f'HISTORY_THROUGH: {ht0}')
    lines.append(f'SEED_LIST_DATE: {sld or ht0}')
    lines.append(f'METHOD: {qualification} / {core_method} / {stream_method}')
    lines.append(f'N47 main streams + {stream_borderline_count} stream-borderline streams saved separately')
    lines.append('')
    for _,r in final.iterrows():
        rank=int(r['playlist_rank']) if str(r['playlist_rank']).strip() else 0
        lines.append(f"{rank:02d}. [{r['playlist_section']}] {r['stream']} | Seed {norm4(r['seed'])} | Core {str(r['core']).zfill(3)} | MBR {str(r['member']).zfill(4)} | Score {float(r['final_plus_transition_x15']):.3f} | StreamRank {int(float(r['stream_gate_rank'])) if str(r['stream_gate_rank']).strip() else ''}")
    (out/'16_MAIN_PLUS_STREAM_BORDERLINE_PLAYLIST.txt').write_text('\n'.join(lines), encoding='utf-8')
    pd.DataFrame([{
        'BUILD':BUILD,'qualification':qualification,'core_method':core_method,'stream_method':stream_method,
        'main_stream_gate':main_stream_gate,'stream_borderline_count':stream_borderline_count,
        'include_stream_borderlines_in_main':include_stream_borderlines_in_main,
        'full_ranked_rows_after_selected_method':len(full_ranked),'main_rows':len(main),'stream_borderline_rows':len(border),'total_displayed_rows':len(final),
        'mirror_filter_used':False,'wide_rescue_pool_used':False,'borderline_definition':'N47+3 stream ranks at stream-gate stage, not wide-pool row fishing'
    }]).to_csv(out/'00_V5_RUN_SUMMARY.csv', index=False)
    return final

def run_date_v5(history_path, profile_dir, out_dir, play_date, history_through=None, seed_list_path=None, exclude_az_md=True, main_stream_gate=47, stream_borderline_count=3, include_stream_borderlines_in_main=False, qualification='major_ge4_and_aggressive', core_method='CORE_max', stream_method='STREAM_all', main_limit=47, winner_targets_path=None):
    out=Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    hist0=read_history(history_path)
    hist, exclusion_audit=apply_exclusions(hist0, exclude_az_md=exclude_az_md)
    play_date=pd.to_datetime(play_date).strftime('%Y-%m-%d')
    if history_through is None:
        history_through=(pd.to_datetime(play_date)-pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    else: history_through=pd.to_datetime(history_through).strftime('%Y-%m-%d')
    prof=load_profiles(profile_dir)
    events=derive_seed_events(hist, history_through, play_date)
    if seed_list_path:
        seed_list=read_seed_list(seed_list_path)
        events=apply_seed_override(events, seed_list)
    full=build_full_enumeration(hist, events, prof)
    gate=build_stream_gate(full, hist=hist, history_through=history_through, mode='baseline_n47')
    q2=build_q2_contenders(full, gate, use_top=int(main_stream_gate)+int(stream_borderline_count))
    step2=apply_step2_transition(q2, hist, history_through, x=15)
    # Write daily ladder files.
    exclusion_audit.to_csv(out/'00_STEP0_EXCLUSION_AUDIT.csv', index=False)
    pd.DataFrame([{'BUILD':BUILD,'history_rows_loaded':len(hist0),'history_rows_after_exclusion':len(hist),'history_through':history_through,'play_date':play_date,'seed_list_uploaded':bool(seed_list_path),'streams':events['stream'].nunique(),'full_enumeration_rows':len(full),'stream_gate_total_used':int(main_stream_gate)+int(stream_borderline_count),'q2_rows':len(q2),'step2_rows':len(step2),'mirror_filter_used':False}]).to_csv(out/'00_RUN_SUMMARY.csv', index=False)
    events.to_csv(out/'01_STEP0_SEEDS_FOR_PLAY_DATE.csv', index=False)
    full.to_csv(out/'02_STEP1_FULL120_FULL_ENUMERATION.csv', index=False)
    gate.to_csv(out/'03_STREAM_GATE_BASELINE_N47_WITH_WINNER_SCALE_AUDIT.csv', index=False)
    q2.to_csv(out/'04_Q2_BASE_CONTENDER_ROWS_CORETOP2_MEMTOP1.csv', index=False)
    step2.to_csv(out/'05_STEP2_CORRECTED_TRANSITION_X15_NO_HARD_DELETE.csv', index=False)
    step3_annot, core_counts, stream_counts=add_rowcount_buckets(step2)
    step3_annot.to_csv(out/'06_STEP3_ROWS_WITH_CORE_STREAM_BUCKETS.csv', index=False)
    core_counts.to_csv(out/'07_STEP3_CORE_ROWCOUNT_MEAN_MEDIAN.csv', index=False)
    stream_counts.to_csv(out/'08_STEP3_STREAM_ROWCOUNT_MEAN_MEDIAN.csv', index=False)
    final=build_v5_manual_playlist_from_step2(step2, out, qualification=qualification, core_method=core_method, stream_method=stream_method, main_stream_gate=main_stream_gate, stream_borderline_count=stream_borderline_count, include_stream_borderlines_in_main=include_stream_borderlines_in_main, main_limit=main_limit)
    # zip
    tag=f"{BUILD}_{play_date}_HISTTHRU_{history_through}".replace('-','')
    zip_path=out.parent/f'{tag}.zip'
    with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file(): z.write(p,p.relative_to(out))
    return zip_path

def main_v5():
    ap=argparse.ArgumentParser(description=BUILD)
    ap.add_argument('--history', required=True)
    ap.add_argument('--profiles', default='profiles')
    ap.add_argument('--out', default=None)
    ap.add_argument('--play-date', required=True)
    ap.add_argument('--history-through', default=None)
    ap.add_argument('--seed-list', default=None)
    ap.add_argument('--include-az-md', action='store_true')
    ap.add_argument('--main-stream-gate', type=int, default=47)
    ap.add_argument('--stream-borderline-count', type=int, default=3)
    ap.add_argument('--include-stream-borderlines-in-main', action='store_true')
    ap.add_argument('--qualification', default='major_ge4_and_aggressive', choices=QUALIFICATION_METHODS)
    ap.add_argument('--core-method', default='CORE_max', choices=CORE_METHODS)
    ap.add_argument('--stream-method', default='STREAM_all', choices=STREAM_METHODS)
    ap.add_argument('--main-limit', type=int, default=47)
    args=ap.parse_args()
    if args.out is None:
        args.out=str(Path('outputs')/f"{BUILD}_{args.play_date.replace('-','')}")
    zp=run_date_v5(args.history,args.profiles,args.out,args.play_date,args.history_through,args.seed_list,exclude_az_md=not args.include_az_md,main_stream_gate=args.main_stream_gate,stream_borderline_count=args.stream_borderline_count,include_stream_borderlines_in_main=args.include_stream_borderlines_in_main,qualification=args.qualification,core_method=args.core_method,stream_method=args.stream_method,main_limit=args.main_limit)
    print(f'Wrote {zp}')

# Use V5 CLI by default.
if __name__=='__main__':
    main_v5()
