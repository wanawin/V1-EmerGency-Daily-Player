from __future__ import annotations
import tempfile, zipfile
from pathlib import Path
import pandas as pd
import streamlit as st

import audit_engine as ae

BUILD_LABEL = 'EMERGENCY DAILY PLAYER V1.5.1 — DOWNLOAD KEY FIX / DECISION LAYER AUDIT / 6 LANES KEPT'

st.set_page_config(page_title='Emergency Daily Player V1.5.1', layout='wide')

ROOT = Path(__file__).resolve().parent
PROFILE_DIR = ROOT / 'profiles'
IN_DIR = ROOT / 'IN'
OUT_ROOT = ROOT / 'outputs'
OUT_ROOT.mkdir(exist_ok=True)


def safe_read_csv(path, **kwargs):
    try:
        p = Path(path)
        if not p.exists() or p.stat().st_size == 0:
            return pd.DataFrame()
        return pd.read_csv(p, **kwargs)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def write_upload(upload, suffix='.csv'):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(upload.getvalue())
    tmp.close()
    return tmp.name


def infer_next_play_date_from_history(path):
    try:
        h = ae.eng.read_history(path)
        if h is None or h.empty or 'date' not in h.columns:
            return pd.to_datetime('2026-06-19').date(), pd.to_datetime('2026-06-18').date()
        mx = pd.to_datetime(h['date'], errors='coerce').max()
        if pd.isna(mx):
            return pd.to_datetime('2026-06-19').date(), pd.to_datetime('2026-06-18').date()
        return (mx + pd.Timedelta(days=1)).date(), mx.date()
    except Exception:
        return pd.to_datetime('2026-06-19').date(), pd.to_datetime('2026-06-18').date()


st.title('Emergency Daily Player V1.5.1')
st.caption(f'BUILD: {BUILD_LABEL}')
st.info('This daily-play version keeps the V5 lanes separate for comparison, adds a Top-30 stream-overlap audit, renames the old “final playlist,” and adds a decision-layer strategy audit while keeping all 6 lanes so it is not confused with an authoritative final selector.')

with st.sidebar:
    st.header('Daily inputs')
    history_file = st.file_uploader('History CSV/TXT through yesterday', type=['csv','txt'])
    use_sample = st.checkbox('Use included 06/18 sample history for 06/19 test', value=(history_file is None))

    if use_sample and history_file is None:
        default_hist_path = str(IN_DIR / 'sample_history_THROUGH_2026-06-18.csv')
        inferred_play, inferred_through = pd.to_datetime('2026-06-19').date(), pd.to_datetime('2026-06-18').date()
    elif history_file is not None:
        # We cannot infer until button time without writing upload; use current fallback in UI.
        default_hist_path = None
        inferred_play, inferred_through = pd.Timestamp.today().date(), (pd.Timestamp.today() - pd.Timedelta(days=1)).date()
    else:
        default_hist_path = None
        inferred_play, inferred_through = pd.to_datetime('2026-06-19').date(), pd.to_datetime('2026-06-18').date()

    st.header('Date controls')
    play_date = st.date_input('PLAY_DATE', value=inferred_play)
    auto_history_through = st.checkbox('Auto history-through = day before play date', value=True)
    history_through = st.date_input('HISTORY_THROUGH', value=(pd.to_datetime(play_date) - pd.Timedelta(days=1)).date(), disabled=auto_history_through)

    st.header('Locked V5 defaults')
    exclude_az_md = st.checkbox('Exclude AZ/MD before seed build', value=True)
    step2_scope = st.selectbox('Step 2 scope', ae.STEP2_SCOPE_OPTIONS, index=0)
    bucket_basis = st.selectbox('Bucket basis', ae.BUCKET_BASIS_OPTIONS, index=0)
    gate_top = st.number_input('Stream gate count', min_value=1, max_value=78, value=50, step=1)
    play_cap = st.number_input('Final play cap', min_value=5, max_value=100, value=50, step=5)
    lane_top_n = st.number_input('Max rows to take from each lane before merge', min_value=1, max_value=100, value=50, step=5)
    write_step2 = st.checkbox('Write full Step2 audit table', value=True)

    st.caption('Defaults match the V5 emergency result: watched8_all_members, final_x15_positive, stream gate 50, cap 50, AZ/MD excluded.')
    run_btn = st.button('Build today’s emergency playlist', type='primary')

st.subheader('What this daily player does')
st.markdown('''
1. Loads history through **HISTORY_THROUGH** only.
2. Builds the watched-8 member candidate universe.
3. Applies the corrected transition x15 and row-count buckets.
4. Runs the fixed V5 emergency lanes that produced the 5-of-7 low-row result.
5. Exports every lane as its own separate playable list for comparison.
6. Also creates a secondary merged-candidate list ranked by best lane rank, not consensus count.
7. Builds a Top-30 stream-overlap audit showing streams that appear across multiple lane tops.
8. Builds a decision-layer strategy audit comparing practical play strategies while keeping all 6 lanes.

This app does **not** use winner files and does **not** use future results.
''')

st.subheader('Default emergency lanes')
st.dataframe(pd.DataFrame(ae.EMERGENCY_LANES), use_container_width=True, height=260)

if run_btn:
    try:
        if use_sample and history_file is None:
            hist_path = str(IN_DIR / 'sample_history_THROUGH_2026-06-18.csv')
        else:
            if history_file is None:
                st.error('Upload a history file or use the included sample.')
                st.stop()
            hist_path = write_upload(history_file, Path(history_file.name).suffix or '.csv')

        ht = '' if auto_history_through else str(history_through)
        tag = f"{ae.EMERGENCY_BUILD_ID}_{pd.to_datetime(play_date).strftime('%Y%m%d')}_{pd.Timestamp.now().strftime('%H%M%S')}"
        out_dir = OUT_ROOT / tag
        progress = st.progress(0)
        status = st.empty()
        status.write('Building daily playlist...')
        with st.spinner('Running Step 0 → Step 2 → V5 emergency lanes...'):
            res = ae.build_emergency_daily_playlist(
                history_path=hist_path,
                profile_dir=str(PROFILE_DIR),
                out_dir=str(out_dir),
                play_date=str(play_date),
                history_through=ht,
                exclude_az_md=bool(exclude_az_md),
                step2_scope=step2_scope,
                bucket_basis=bucket_basis,
                gate_top=int(gate_top),
                play_cap=int(play_cap),
                lane_top_n=int(lane_top_n),
                write_full_step2=bool(write_step2),
            )
        progress.progress(1.0)
        status.success('Playlist built.')
        st.session_state['daily_result'] = res
    except Exception as e:
        st.exception(e)

res = st.session_state.get('daily_result')
if res:
    st.header('Daily playlist results')
    summary = res.get('summary', pd.DataFrame())
    final = res.get('merged_candidates', res.get('final', pd.DataFrame()))
    overlap_streams = res.get('stream_overlap_streams', pd.DataFrame())
    overlap_rows = res.get('stream_overlap_rows', pd.DataFrame())
    lanes = res.get('lane_summary', pd.DataFrame())
    lane_rows = res.get('lane_rows', pd.DataFrame())

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    if not summary.empty:
        s = summary.iloc[0]
        c1.metric('Merged candidate rows', int(s.get('merged_candidate_rows', s.get('final_rows', 0))))
        c2.metric('Step2 rows', int(s.get('step2_rows', 0)))
        c3.metric('Unique lane rows before cap', int(s.get('unique_lane_rows_before_cap', 0)))
        c4.metric('Play cap', int(s.get('play_cap', 0)))
        c5.metric('Profile files loaded', int(s.get('profile_files_nonempty', 0)))
        c6.metric('Support rows', int(s.get('support_major_nonzero_step2', 0)))
        if int(s.get('merged_candidate_rows', s.get('final_rows', 0))) == 0:
            st.error('Final rows are 0. This is not a valid daily output. Most likely the profile/support CSV files are missing from the deployed repo. Upload the profiles/ folder from the zip, or place the V6_8CORE profile CSVs at the repo root.')

    tabs = st.tabs(['Merged candidates by best lane rank','Stream overlap Top30','Decision layer strategies','Lane summary','Separate lane rows','All lane rows before dedupe','Run summary','Downloads'])
    with tabs[0]:
        st.caption('Renamed from FINAL PLAYLIST. This is a merged lane candidate list, not the authoritative final selector.')
        if final is None or final.empty:
            st.warning('No playable rows were produced by the current emergency lanes.')
        else:
            q = st.text_input('Search combined list by stream/core/member/seed', '')
            show = final.copy()
            if q:
                show = show[show.astype(str).agg(' '.join, axis=1).str.lower().str.contains(q.lower(), na=False)]
            st.dataframe(show, use_container_width=True, height=560)
    with tabs[1]:
        st.caption('Top 30 from each lane, grouped by stream. This tests stream agreement without requiring exact core/member overlap.')
        if overlap_streams is None or overlap_streams.empty:
            st.info('No stream-overlap rows to preview.')
        else:
            st.dataframe(overlap_streams, use_container_width=True, height=360)
            st.caption('Rows behind the stream-overlap audit')
            st.dataframe(overlap_rows.head(1500), use_container_width=True, height=360)
    with tabs[2]:
        outdir = Path(res['out_dir'])
        p = outdir / '07_DECISION_LAYER_STRATEGY_SUMMARY.csv'
        dec = safe_read_csv(p) if p.exists() else pd.DataFrame()
        st.caption('Compares practical candidate strategies without deleting any of the 6 lanes.')
        st.dataframe(dec, use_container_width=True, height=320)
        if p.exists():
            st.download_button('Download 07_DECISION_LAYER_STRATEGY_SUMMARY.csv', p.read_bytes(), file_name='07_DECISION_LAYER_STRATEGY_SUMMARY.csv', mime='text/csv', key='dl_decision_summary_tab')
    with tabs[3]:
        st.dataframe(lanes, use_container_width=True, height=360)
    with tabs[4]:
        if lane_rows is None or lane_rows.empty:
            st.info('No separate lane rows to preview.')
        else:
            lane_options = ['ALL'] + sorted(lane_rows['lane_id'].astype(str).unique().tolist()) if 'lane_id' in lane_rows.columns else ['ALL']
            chosen_lane = st.selectbox('Choose lane to compare', lane_options)
            ql = st.text_input('Search selected lane by stream/core/member/seed', '')
            show_lane = lane_rows.copy()
            if chosen_lane != 'ALL' and 'lane_id' in show_lane.columns:
                show_lane = show_lane[show_lane['lane_id'].astype(str) == chosen_lane]
            if ql:
                show_lane = show_lane[show_lane.astype(str).agg(' '.join, axis=1).str.lower().str.contains(ql.lower(), na=False)]
            st.dataframe(show_lane.head(1500), use_container_width=True, height=560)
    with tabs[5]:
        if lane_rows is None or lane_rows.empty:
            st.info('No lane rows to preview.')
        else:
            q2 = st.text_input('Search all lane rows by stream/core/member/seed/lane', '')
            show2 = lane_rows.copy()
            if q2:
                show2 = show2[show2.astype(str).agg(' '.join, axis=1).str.lower().str.contains(q2.lower(), na=False)]
            st.dataframe(show2.head(1500), use_container_width=True, height=560)
    with tabs[6]:
        st.dataframe(summary, use_container_width=True, height=220)
    with tabs[7]:
        outdir = Path(res['out_dir'])
        zip_path = Path(res['zip_path'])
        if zip_path.exists():
            st.download_button('Download full daily output ZIP', zip_path.read_bytes(), file_name=zip_path.name, mime='application/zip', key='dl_full_zip')
        for name, mime in [
            ('02_MERGED_LANE_CANDIDATES_PRINTABLE.txt','text/plain'),
            ('02_MERGED_LANE_CANDIDATES_BY_BEST_LANE_RANK.csv','text/csv'),
            ('06A_STREAM_OVERLAP_TOP30_STREAMS.csv','text/csv'),
            ('06B_STREAM_OVERLAP_TOP30_ROWS.csv','text/csv'),
            ('06_STREAM_OVERLAP_TOP30_PRINTABLE.txt','text/plain'),
            ('07_DECISION_LAYER_STRATEGY_SUMMARY.csv','text/csv'),
            ('07_DECISION_LAYER_STRATEGY_SUMMARY.txt','text/plain'),
            ('02A_COMBINED_BY_BEST_LANE_RANK.csv','text/csv'),
            ('02B_COMBINED_BY_CONSENSUS_LEGACY_DO_NOT_USE_AS_PRIMARY.csv','text/csv'),
            ('01_LANE_SUMMARY.csv','text/csv'),
            ('03_ALL_LANE_ROWS_BEFORE_DEDUPE.csv','text/csv'),
            ('04_FULL_STEP2_BUCKETED_ROWS.csv','text/csv'),
            ('00_RUN_SUMMARY.csv','text/csv'),
        ]:
            p = outdir / name
            if p.exists():
                st.download_button(f'Download {name}', p.read_bytes(), file_name=name, mime=mime, key=f'dl_file_{name}')
