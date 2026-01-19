
# app.py
import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update, ALL
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objs as go
import base64
import io
import json
import warnings
import os
import sys

# 忽略openpyxl样式警告
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

# ========== 计算 base_path（支持本地/云端）并固定 assets 路径 ==========
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    BASE_PATH = sys._MEIPASS  # 给未来 EXE 预留；在 Render 在线运行时不会触发
else:
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
ASSETS_PATH = os.path.join(BASE_PATH, "assets")

# 初始化全局变量
record_df = pd.DataFrame()
step_df = pd.DataFrame()
global_start = None
global_end = None
global_duration = 0
cycle_info = []
step_info = []

# 常量
current = 'Current(A)'
voltage = 'Voltage(V)'
energy = 'Energy(Wh)'
power = 'Power(W)'
v_cols = ['V1(V)', 'V2(V)', 'V3(V)', 'V4(V)']
soc_col = 'SOC(%)'
t_col = ['T1(℃)', 'T2(℃)', 'T3(℃)', 'T4(℃)']
fault_col = ['Fault_State_1', 'Fault_State_2', 'Fault_State_3', 'Fault_State_4',
             'Fault_State_5', 'Fault_State_6']

# 最小像素宽度保障（避免过窄时看不见条）
MIN_CYCLE_PX = 30
MIN_STEP_PX = 10

# Dash App（指定 assets_folder）
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], assets_folder=ASSETS_PATH)
app.config.suppress_callback_exceptions = True
# ✅ 允许带 allow_duplicate 的回调在初始时合法触发
app.config.prevent_initial_callbacks = 'initial_duplicate'

# ✅ 暴露 Flask server 给 gunicorn（Render/Heroku 会用到）
server = app.server


# ------------------ 工具函数：统一列名/只认 Date 时间列 ------------------
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _ensure_date_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    只认 'Date' 列为时间轴。若存在则转换为 datetime；否则保持现状（后续会提示）。
    """
    if df is None or df.empty:
        return df
    df = _normalize_columns(df)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    return df


# ------------------ 绘图函数（固定 8 张） ------------------
def _empty_fig(title):
    fig = go.Figure()
    fig.update_layout(title=title, width=300, height=189,
                      margin=dict(l=30, r=10, t=40, b=30))
    return fig

def plot_current(df):
    if df.empty or 'Date' not in df or current not in df:
        return _empty_fig("Current[A]")
    return go.Figure([
        go.Scatter(x=df['Date'], y=df[current], mode='lines', name=current)
    ]).update_layout(title="Current[A]", width=300, height=189,
                     margin=dict(l=30, r=10, t=40, b=30), uirevision='current')

def plot_v0(df):
    if df.empty or 'Date' not in df or voltage not in df:
        return _empty_fig("Voltage[V]")
    return go.Figure([
        go.Scatter(x=df['Date'], y=df[voltage], mode='lines', name=voltage)
    ]).update_layout(title="Voltage[V]", width=300, height=189,
                     margin=dict(l=30, r=10, t=40, b=30), uirevision='v0')

def plot_energy(df):
    if df.empty or 'Date' not in df or energy not in df:
        return _empty_fig("Energy[Wh]")
    return go.Figure([
        go.Scatter(x=df['Date'], y=df[energy], mode='lines', name=energy)
    ]).update_layout(title="Energy[Wh]", width=300, height=189,
                     margin=dict(l=30, r=10, t=40, b=30), uirevision='energy')

def plot_power(df):
    if df.empty or 'Date' not in df or power not in df:
        return _empty_fig("Power[W]")
    return go.Figure([
        go.Scatter(x=df['Date'], y=df[power], mode='lines', name=power)
    ]).update_layout(title="Power[W]", width=300, height=189,
                     margin=dict(l=30, r=10, t=40, b=30), uirevision='power')

def plot_soc(df):
    if df.empty or 'Date' not in df or soc_col not in df:
        return _empty_fig("SOC[%]")
    return go.Figure([
        go.Scatter(x=df['Date'], y=df[soc_col], mode='lines', name='SOC(%)')
    ]).update_layout(title="SOC[%]", width=300, height=189,
                     margin=dict(l=30, r=10, t=40, b=30), uirevision='soc')

def plot_voltage(df):
    fig = go.Figure()
    if not df.empty and 'Date' in df:
        for col in v_cols:
            if col in df.columns:
                fig.add_trace(go.Scatter(x=df['Date'], y=df[col], mode='lines', name=col))
    fig.update_layout(title="Voltage[V]", width=300, height=189,
                      margin=dict(l=30, r=10, t=40, b=30),
                      legend=dict(font=dict(size=8), itemsizing="constant",
                                  x=1.02, y=1, xanchor="left", yanchor="top"),
                      uirevision='voltage')
    return fig

def plot_temp(df):
    fig = go.Figure()
    if not df.empty and 'Date' in df:
        for temp in t_col:
            if temp in df.columns:
                fig.add_trace(go.Scatter(x=df['Date'], y=df[temp], mode='lines', name=temp))
    fig.update_layout(title="Temperature[℃]", width=300, height=189,
                      margin=dict(l=30, r=10, t=40, b=30),
                      legend=dict(font=dict(size=8), itemsizing="constant",
                                  x=1.02, y=1, xanchor="left", yanchor="top"),
                      uirevision='temp')
    return fig

def plot_fault(df):
    fig = go.Figure()
    if not df.empty and 'Date' in df:
        for fault in fault_col:
            if fault in df.columns:
                fig.add_trace(go.Scatter(x=df['Date'], y=df[fault], mode='lines', name=fault))
    fig.update_layout(title="Fault", width=300, height=189,
                      margin=dict(l=30, r=10, t=40, b=30),
                      legend=dict(font=dict(size=8), itemsizing="constant",
                                  x=1.02, y=1, xanchor="left", yanchor="top"),
                      uirevision='fault')
    return fig


# ------------------ Slicer 组件生成（百分比布局） ------------------
def create_slicer_bars(selected_id=None):
    def _build_cycle_bar_for_global(c, is_selected, order_idx, n_cycles):
        if global_duration and global_duration > 0:
            width_pct = (c['duration'] / global_duration) * 100.0
            left_pct = ((c['start'] - global_start).total_seconds() / global_duration) * 100.0
        else:
            width_pct = 100.0 / max(n_cycles, 1)
            left_pct = order_idx * width_pct
        return {
            'label': f"Cycle {c['cycle']}",
            'idx': f"cycle-{c['cycle']}",
            'width': width_pct,
            'left': left_pct,
            'cycle': c,
            'selected': is_selected
        }

    def _build_step_bars_for_global(cbar):
        bars = []
        c = cbar['cycle']
        c_start = c['start']; c_end = c['end']
        c_duration = max((c_end - c_start).total_seconds(), 0)
        c_left = cbar['left']; c_width = cbar['width']

        steps = [s for s in step_info if s['cycle'] == c['cycle']]
        steps = sorted(steps, key=lambda x: x['start'])
        if not steps:
            return bars

        if c_duration <= 0:
            n = len(steps)
            each = c_width / n if n else c_width
            acc_left = c_left
            for i, s in enumerate(steps):
                width_pct = each if i < n - 1 else (c_left + c_width - acc_left)
                bars.append({
                    'label': f"Step {s['step']}",
                    'idx': f"step-{s['id']}",
                    'width': max(width_pct, 0),
                    'left': acc_left,
                    'cycle': c['cycle'],
                    'step': s['step'],
                    'selected': (selected_id == f"step-{s['id']}")
                })
                acc_left += width_pct
            return bars

        prev_right = c_left
        n = len(steps)
        for i, s in enumerate(steps):
            rel_left = max((s['start'] - c_start).total_seconds() / c_duration, 0.0)
            if i == n - 1:
                left_pct = prev_right
                right_pct = c_left + c_width
                width_pct = max(right_pct - left_pct, 0.0)
            else:
                rel_right = max((s['end'] - c_start).total_seconds() / c_duration, rel_left)
                left_pct = c_left + rel_left * c_width
                right_pct = c_left + rel_right * c_width
                width_pct = max(right_pct - left_pct, 0.0)

            bars.append({
                'label': f"Step {s['step']}",
                'idx': f"step-{s['id']}",
                'width': width_pct,
                'left': left_pct,
                'cycle': c['cycle'],
                'step': s['step'],
                'selected': (selected_id == f"step-{s['id']}")
            })
            prev_right = left_pct + width_pct

        return bars

    def _build_cycle_bar_full_width(c, is_selected):
        return {'label': f"Cycle {c['cycle']}", 'idx': f"cycle-{c['cycle']}",
                'width': 100.0, 'left': 0.0, 'cycle': c, 'selected': is_selected}

    def _build_step_bars_fill_cycle(c):
        bars = []
        steps = [s for s in step_info if s['cycle'] == c['cycle']]
        steps = sorted(steps, key=lambda x: x['start'])
        c_start = c['start']; c_end = c['end']
        c_duration = max((c_end - c_start).total_seconds(), 0.0)
        if not steps:
            return bars
        if c_duration <= 0:
            s0 = steps[0]
            bars.append({'label': f"Step {s0['step']}", 'idx': f"step-{s0['id']}",
                         'width': 100.0, 'left': 0.0, 'cycle': c['cycle'],
                         'step': s0['step'], 'selected': (selected_id == f"step-{s0['id']}")})
            return bars

        prev_right = 0.0
        n = len(steps)
        for i, s in enumerate(steps):
            if i == n - 1:
                left_pct = prev_right; width_pct = max(100.0 - left_pct, 0.0)
            else:
                rel_left = max((s['start'] - c_start).total_seconds() / c_duration, 0.0)
                rel_right = max((s['end'] - c_start).total_seconds() / c_duration, rel_left)
                left_pct = max(prev_right, rel_left * 100.0)
                right_pct = rel_right * 100.0
                if right_pct < left_pct: right_pct = left_pct
                width_pct = max(right_pct - left_pct, 0.0)

            bars.append({'label': f"Step {s['step']}", 'idx': f"step-{s['id']}",
                         'width': width_pct, 'left': left_pct, 'cycle': c['cycle'],
                         'step': s['step'], 'selected': (selected_id == f"step-{s['id']}")})
            prev_right = left_pct + width_pct

        if bars:
            end_gap = 100.0 - (bars[-1]['left'] + bars[-1]['width'])
            if abs(end_gap) > 1e-6:
                bars[-1]['width'] = max(bars[-1]['width'] + end_gap, 0.0)
        return bars

    global_selected = (selected_id == 'global')

    is_cycle_selected = False
    selected_cycle_num = None
    if selected_id and isinstance(selected_id, str):
        if selected_id.startswith('cycle-'):
            try:
                selected_cycle_num = int(selected_id.split('-')[1]); is_cycle_selected = True
            except Exception:
                is_cycle_selected = False
        elif selected_id.startswith('step-'):
            try:
                _, cnum, _ = selected_id.split('-'); selected_cycle_num = int(cnum); is_cycle_selected = True
            except Exception:
                is_cycle_selected = False

    cycle_bars = []
    if is_cycle_selected:
        c_list = [c for c in cycle_info if c['cycle'] == selected_cycle_num]
        if c_list:
            c = c_list[0]; cycle_bars.append(_build_cycle_bar_full_width(c, True))
        else:
            is_cycle_selected = False

    if not is_cycle_selected:
        n_cycles = len(cycle_info) if cycle_info else 1
        for idx, c in enumerate(cycle_info):
            is_selected = (selected_id == f"cycle-{c['cycle']}")
            cycle_bars.append(_build_cycle_bar_for_global(c, is_selected, idx, n_cycles))

    step_bars = []
    if is_cycle_selected and cycle_bars:
        c = cycle_bars[0]['cycle']; step_bars = _build_step_bars_fill_cycle(c)
    else:
        for cbar in cycle_bars:
            step_bars.extend(_build_step_bars_for_global(cbar))

    def _pct(x):
        return f"{max(min(x, 100.0), 0.0):.6f}%"

    slicer_components = [
        html.Div("Slicer：", style={'fontWeight': 'bold'}),

        # Global
        html.Div([
            html.Div(
                "Global",
                id={'type': 'slicer', 'index': 'global'},
                n_clicks=0,
                style={
                    'position': 'absolute','left': '0%','width': '100%',
                    'height': '25px','lineHeight': '25px','textAlign': 'center',
                    'background': '#007bff' if global_selected else '#888',
                    'color': '#fff','borderRadius': '6px','cursor': 'pointer',
                    'fontWeight': 'bold','fontSize': '13px','overflow': 'hidden',
                    'border': '1px solid #fff','boxSizing': 'border-box'
                }
            )
        ], style={'width': '100%', 'height': '25px', 'position': 'relative', 'background': '#eee'}),

        html.Br(),

        # Cycle controls
        dbc.Row([
            dbc.Col([
                html.Button("Collapse/Expand Cycles", id="toggle-cycles",
                            n_clicks=0, className="btn btn-sm btn-outline-secondary",
                            style={'marginBottom': '5px'})
            ], width="auto"),
            dbc.Col([
                dcc.Dropdown(
                    id="cycle-select",
                    options=[{'label': f"Cycle {c['cycle']}", 'value': c['cycle']} for c in cycle_info],
                    value=(selected_cycle_num if is_cycle_selected else None),
                    placeholder="Select cycle", clearable=True,
                    persistence=True, persistence_type='memory',
                    style={'minWidth': '220px', 'marginBottom': '5px', 'fontSize': '12px'}
                )
            ], width="auto", className="ms-auto")
        ], justify="between"),

        # Cycle slicers
        html.Div(
            id="cycle-slicers-container",
            children=[html.Div([
                *[html.Div(
                    bar['label'],
                    id={'type': 'slicer', 'index': bar['idx']},
                    n_clicks=0,
                    style={
                        'position': 'absolute','left': _pct(bar["left"]),
                        'width': _pct(bar["width"]),'minWidth': f'{MIN_CYCLE_PX}px',
                        'height': '20px','lineHeight': '20px','textAlign': 'center',
                        'background': '#007bff' if bar['selected'] else '#bbb',
                        'color': '#fff' if bar['selected'] else '#222',
                        'borderRadius': '6px','cursor': 'pointer','fontWeight': 'bold',
                        'fontSize': '13px','overflow': 'hidden','border': '1px solid #fff',
                        'boxSizing': 'border-box','whiteSpace': 'nowrap'
                    }
                ) for bar in cycle_bars]
            ], style={'width': '100%', 'height': '22px', 'position': 'relative', 'background': '#eee', 'overflow': 'hidden'})],
            style={'display': 'block'}
        ),

        html.Br(),

        # Step controls
        dbc.Row([
            dbc.Col([
                html.Button("Collapse/Expand Steps", id="toggle-steps",
                            n_clicks=0, className="btn btn-sm btn-outline-secondary",
                            style={'marginBottom': '5px'})
            ], width="auto")
        ]),

        # Step slicers
        html.Div(
            id="step-slicers-container",
            children=[html.Div([
                *[html.Div(
                    bar['label'],
                    id={'type': 'slicer', 'index': bar['idx']},
                    n_clicks=0,
                    style={
                        'position': 'absolute','left': _pct(bar["left"]),
                        'width': _pct(bar["width"]),'minWidth': f'{MIN_STEP_PX}px',
                        'height': '15px','lineHeight': '15px','textAlign': 'center',
                        'background': '#007bff' if bar['selected'] else '#bbb',
                        'color': '#fff' if bar['selected'] else '#222',
                        'borderRadius': '6px','cursor': 'pointer','fontWeight': 'bold',
                        'fontSize': '13px','overflow': 'hidden','border': '1px solid #fff',
                        'boxSizing': 'border-box','whiteSpace': 'nowrap'
                    }
                ) for bar in step_bars]
            ], style={'width': '100%', 'height': '17px', 'position': 'relative', 'background': '#eee', 'overflow': 'hidden'})],
            style={'display': 'block'}
        )
    ]
    return slicer_components


# ------------------ 数据初始化（健壮处理 step 表列名） ------------------
def init_data(file_path=None, contents=None, filename=None, max_rows=None):
    """
    初始化/更新数据；max_rows 为 int 则 record sheet 仅读取前 max_rows 行。
    对 step 表的 'Cycle Index' / 'Step Index' / 'Onset/Oneset/Start' / 'End' 列做别名兼容。
    """
    global record_df, step_df, global_start, global_end, global_duration, cycle_info, step_info

    try:
        # 读 Excel
        if contents:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            excel_file = pd.ExcelFile(io.BytesIO(decoded), engine="openpyxl")
            record_df = pd.read_excel(excel_file, sheet_name="record", nrows=max_rows)
            step_df = pd.read_excel(excel_file, sheet_name="step")
        elif file_path:
            record_df = pd.read_excel(file_path, sheet_name="record", engine="openpyxl", nrows=max_rows)
            step_df = pd.read_excel(file_path, sheet_name="step", engine="openpyxl")
        else:
            record_df = pd.DataFrame(); step_df = pd.DataFrame()

        # 规范 Date
        if not record_df.empty:
            record_df = _ensure_date_only(record_df)
            if 'Date' in record_df.columns and record_df['Date'].notna().any():
                global_start = pd.to_datetime(record_df['Date']).min()
                global_end = pd.to_datetime(record_df['Date']).max()
                global_duration = (global_end - global_start).total_seconds()
            else:
                global_start = pd.Timestamp.now(); global_end = pd.Timestamp.now(); global_duration = 0
        else:
            global_start = pd.Timestamp.now(); global_end = pd.Timestamp.now(); global_duration = 0

        # 组装 step/cycle（健壮）
        cycle_info.clear(); step_info.clear()
        if not step_df.empty:
            step_df = _normalize_columns(step_df)
            cols = set(step_df.columns)

            # 列别名解析
            def pick(col_alias_list):
                for c in col_alias_list:
                    if c in cols:
                        return c
                return None

            cycle_col = pick(['Cycle Index', 'Cycle', 'CycleIndex'])
            step_col  = pick(['Step Index', 'Step', 'StepIndex'])

            start_col = pick(['Oneset Date', 'Onset Date', 'Start Date', 'Start Time', 'Begin Time'])
            end_col   = pick(['End Date', 'End Time', 'Stop Time'])

            # cycle_info
            if cycle_col and start_col and end_col:
                for cycle in step_df[cycle_col].dropna().unique():
                    steps = step_df[step_df[cycle_col] == cycle]
                    if not steps.empty:
                        start = pd.to_datetime(steps[start_col], errors='coerce').min()
                        end   = pd.to_datetime(steps[end_col],   errors='coerce').max()
                        duration = (end - start).total_seconds() if pd.notna(start) and pd.notna(end) else 0
                        try:
                            cyc_int = int(cycle)
                        except Exception:
                            cyc_int = cycle
                        cycle_info.append({'cycle': cyc_int, 'start': start, 'end': end, 'duration': duration})

            # step_info
            if cycle_col and step_col and start_col and end_col:
                for _, row in step_df.iterrows():
                    s = pd.to_datetime(row.get(start_col), errors='coerce')
                    e = pd.to_datetime(row.get(end_col),   errors='coerce')
                    cyc = row.get(cycle_col); stp = row.get(step_col)
                    if pd.notna(s) and pd.notna(e):
                        try: cyc_i = int(cyc)
                        except Exception: cyc_i = cyc
                        try: stp_i = int(stp)
                        except Exception: stp_i = stp
                        step_info.append({
                            'cycle': cyc_i, 'step': stp_i, 'start': s, 'end': e,
                            'id': f"{cyc_i}-{stp_i}", 'duration': (e - s).total_seconds()
                        })

        return True, filename if filename else file_path
    except Exception as e:
        print(f"Fail to load: {e}")
        # 降级为空，避免应用崩溃
        record_df = pd.DataFrame(); step_df = pd.DataFrame()
        global_start = pd.Timestamp.now(); global_end = pd.Timestamp.now(); global_duration = 0
        cycle_info.clear(); step_info.clear()
        return False, str(e)


# 初始加载示例（快速模式：仅前 500 行）
DEFAULT_SAMPLE = os.path.join(BASE_PATH, "sample_data", "Charge and discharge 2 cycles_BTS85-86-1-1-104.xlsx")
if os.path.exists(DEFAULT_SAMPLE):
    init_data(DEFAULT_SAMPLE, max_rows=500)
else:
    # 无样例文件也不影响运行；可通过上传功能载入
    pass


# ------------------ 小图 + 放大按钮 ------------------
def _graph_cell(graph_id, expand_index):
    return html.Div([
        html.Button(
            "⤢", id={'type': 'expand', 'index': expand_index}, n_clicks=0, title="放大 / Focus",
            style={
                'position': 'absolute', 'right': '6px', 'top': '6px',
                'zIndex': 2, 'padding': '2px 6px',
                'backgroundColor': 'rgba(255,255,255,0.9)', 'border': '1px solid #999',
                'borderRadius': '4px', 'cursor': 'pointer'
            }
        ),
        dcc.Graph(id=graph_id, style={'width': '189px', 'height': '189px'},
                  config={'displaylogo': False})
    ], style={'position': 'relative'})


# ------------------ App 布局 ------------------
app.layout = dbc.Container([
    dcc.Store(id='selected-slicer', data='global'),
    dcc.Store(id='xaxis-range', data={}),
    dcc.Store(id='focused-graph', data=None),
    dcc.Store(id='calendar-range', data=None),

    # Layout Setting 相关
    dcc.Store(id='layout-staging', data=[]),
    dcc.Store(id='layout-applied', data=[]),
    dcc.Store(id='layout-id-seq', data=0),
    dcc.Download(id='download-layout'),

    # 顶部标题 + 日期（加入浅灰底 + 圆角 + 细边框 + assets/Logo.png 圆角）
    dbc.Row(
        [
            # 左侧：Logo + 标题
            dbc.Col(
                html.Div(
                    [
                        html.Img(
                            src=app.get_asset_url("Logo.png"),
                            style={
                                'height': '100px', 'width': '100px',
                                'borderRadius': '8px',
                                'objectFit': 'cover',
                                'display': 'block'
                            }
                        ),
                        html.H3("Battery Monkey", className="mb-0", style={'marginLeft': '10px'})
                    ],
                    style={'display': 'flex', 'alignItems': 'center'}
                ),
                width="auto"
            ),

            # 右侧：按钮 + 日期
            dbc.Col(
                html.Div(
                    [
                        dbc.Button("Layout Setting", id="layout-btn",
                                   color="secondary", outline=True, size="sm",
                                   style={'marginRight': '10px'}),
                        dcc.DatePickerRange(
                            id='date-range', start_date=None, end_date=None,
                            display_format='YYYY-MM-DD', clearable=True,
                            minimum_nights=0, with_portal=True,
                            number_of_months_shown=1, day_size=28,
                            style={'width': '200px','fontSize': '12px',
                                   'transform': 'scale(0.70)','transformOrigin': 'right center'}
                        ),
                    ],
                    style={'display': 'flex','alignItems': 'center','justifyContent': 'flex-end','gap': '8px'}
                ),
                width=True, className="d-flex align-items-center justify-content-end"
            )
        ],
        align="center",
        style={
            'background': '#f6f7f8',
            'border': '1px solid #e6e6e6',
            'borderRadius': '10px',
            'padding': '8px 12px',
            'marginBottom': '10px'
        }
    ),

    # 文件上传 + 快速模式
    dbc.Row([
        dbc.Col([
            dcc.Upload(
                id='upload-data',
                children=html.Div(['Drag and drop or ', html.A('Select file')]),
                style={
                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                    'borderWidth': '1px', 'borderStyle': 'dashed',
                    'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px 0'
                },
                accept='.xlsx', multiple=False
            ),
        ], width=8),
        dbc.Col([
            dbc.Switch(id='fast-load', value=False,
                       label='Load first 500 rows (fast test)', className='mb-2'),
            html.Div(id="file-name", style={'marginTop': '10px'}),
            html.Div(id="upload-status", style={'marginTop': '5px', 'color': 'red'})
        ], width=4)
    ]),

    # 放大视图容器
    html.Div(id="focused-view", style={'marginBottom': '10px'}),

    # 固定 8 张图
    html.Div(
        id="charts-grid-container",
        children=[
            dbc.Row([
                dbc.Col(_graph_cell("current-graph", "current"), width=3),
                dbc.Col(_graph_cell("v0-graph", "v0"), width=3),
                dbc.Col(_graph_cell("energy-graph", "energy"), width=3),
                dbc.Col(_graph_cell("power-graph", "power"), width=3),
            ], justify="start"),
            dbc.Row([
                dbc.Col(_graph_cell("soc-graph", "soc"), width=3),
                dbc.Col(_graph_cell("volt-graph", "volt"), width=3),
                dbc.Col(_graph_cell("temp-graph", "temp"), width=3),
                dbc.Col(_graph_cell("fault-graph", "fault"), width=3),
            ], justify="start"),
        ]
    ),

    # 自定义图区域
    html.H5("Custom Charts", style={'marginTop': '12px'}),
    html.Div(
        [
            html.Span("Applied layout: ", className="text-muted"),
            html.Span(id="applied-count-badge", className="badge bg-secondary"),
            html.Span(id="applied-brief", className="ms-2 text-muted", style={'fontSize': '12px'})
        ],
        className="mb-2"
    ),
    html.Div(id="custom-charts-container", children=[]),

    html.Hr(),

    # Slicer
    html.Div(id="slicer-container", children=create_slicer_bars('global'), style={'marginTop': '20px', 'width': '100%'}),

    # Layout Setting Modal
    dbc.Modal(
        id="layout-modal", is_open=False, size="lg", backdrop=True,
        children=[
            dbc.ModalHeader(
                dbc.Row(
                    [
                        dbc.Col(html.H5("Layout Setting", className="mb-0"), width="auto"),
                        dbc.Col(
                            html.Button("×", id="layout-close", n_clicks=0,
                                        className="btn btn-link", title="Close",
                                        style={'fontSize': '20px','lineHeight': '1',
                                               'textDecoration': 'none','color': '#6c757d'}),
                            width="auto", className="ms-auto"
                        )
                    ],
                    align="center", className="g-2"
                )
            ),
            dbc.ModalBody(
                [
                    html.Div(id="layout-editor", children=[]),
                    html.Hr(),
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Upload(
                                    id='upload-layout',
                                    children=dbc.Button("Load layout", id="load-layout-btn",
                                                        outline=True, color="secondary", size="sm"),
                                    accept='.csv', multiple=False
                                ),
                                width="auto"
                            ),
                            dbc.Col(
                                dbc.Button("Save layout", id="save-layout-btn",
                                           outline=True, color="secondary", size="sm"),
                                width="auto"
                            ),
                        ],
                        className="mb-2 g-2"
                    ),
                ],
                style={'maxHeight': '60vh', 'overflowY': 'auto'}
            ),
            dbc.ModalFooter(
                [
                    dbc.Button("Cancel", id="layout-cancel", outline=True, color="secondary", className="me-2"),
                    dbc.Button("Apply", id="layout-apply", color="primary")
                ]
            )
        ]
    ),
], fluid=True)


# ------------------ 上传文件回调 ------------------
@app.callback(
    [Output('file-name', 'children'),
     Output('upload-status', 'children'),
     Output('slicer-container', 'children'),
     Output('selected-slicer', 'data'),
     Output('focused-graph', 'data')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('fast-load', 'value'),
    prevent_initial_call=False
)
def handle_file_upload(contents, filename, fast_load):
    if contents is None and filename is None:
        return "File: No file", "", create_slicer_bars('global'), 'global', None

    max_rows = 500 if fast_load else None
    success, msg = init_data(contents=contents, filename=filename, max_rows=max_rows)
    if success:
        slicer_bars = create_slicer_bars('global')
        file_info = f"File: {filename}" if filename else "File: loaded"
        if fast_load:
            file_info += "  (fast: first 500 rows)"

        status_msgs = []
        if record_df.empty or 'Date' not in record_df.columns or not record_df['Date'].notna().any():
            status_msgs.append("Warning: 'Date' column is missing or cannot be parsed. Charts will be empty.")
        # Step 列缺失提示（不阻断）
        step_cols = set(step_df.columns) if not step_df.empty else set()
        required_any = any(k in step_cols for k in ['Cycle Index','Cycle','CycleIndex'])
        required_any = required_any and any(k in step_cols for k in ['Step Index','Step','StepIndex'])
        required_any = required_any and any(k in step_cols for k in ['Oneset Date','Onset Date','Start Date','Start Time','Begin Time'])
        required_any = required_any and any(k in step_cols for k in ['End Date','End Time','Stop Time'])
        if not step_df.empty and not required_any:
            status_msgs.append("Note: 'step' sheet does not contain recognizable cycle/step/start/end columns; slicers will show Global only.")

        return file_info, " ".join(status_msgs), slicer_bars, 'global', None
    else:
        return "Error", f"Error: {msg}", create_slicer_bars('global'), 'global', None


# ------------------ 切换 Cycle/Step 显示 ------------------
@app.callback(
    Output('cycle-slicers-container', 'style'),
    Input('toggle-cycles', 'n_clicks'),
    State('cycle-slicers-container', 'style'),
    prevent_initial_call=True
)
def toggle_cycles(n_clicks, current_style):
    if current_style and current_style.get('display') == 'block':
        return {'display': 'none'}
    return {'display': 'block'}

@app.callback(
    Output('step-slicers-container', 'style'),
    Input('toggle-steps', 'n_clicks'),
    State('step-slicers-container', 'style'),
    prevent_initial_call=True
)
def toggle_steps(n_clicks, current_style):
    if current_style and current_style.get('display') == 'block':
        return {'display': 'none'}
    return {'display': 'block'}


# ------------------ 日历范围 -> Store ------------------
@app.callback(
    [Output('calendar-range', 'data'),
     Output('date-range', 'start_date'),
     Output('date-range', 'end_date')],
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date')],
    prevent_initial_call=False
)
def update_calendar_store(start_date, end_date):
    if not start_date or not end_date:
        return (None, start_date, end_date)
    try:
        st = pd.to_datetime(f"{start_date} 00:00:00")
        et = pd.to_datetime(f"{end_date} 23:59:59")
        if st > et: st, et = et, st
        return ({'start': st.isoformat(), 'end': et.isoformat()}, start_date, end_date)
    except Exception:
        return (None, start_date, end_date)


# ------------------ 固定 8 图更新 + Slicer 同步（含 allow_duplicate） ------------------
@app.callback(
    [
        Output("current-graph", "figure"),
        Output("v0-graph", "figure"),
        Output("energy-graph", "figure"),
        Output("power-graph", "figure"),
        Output("soc-graph", "figure"),
        Output("volt-graph", "figure"),
        Output("temp-graph", "figure"),
        Output("fault-graph", "figure"),
        Output("slicer-container", "children", allow_duplicate=True),   # ✅
        Output("selected-slicer", "data", allow_duplicate=True),        # ✅
        Output("cycle-select", "value", allow_duplicate=True),          # ✅
    ],
    [
        Input({'type': 'slicer', 'index': ALL}, 'n_clicks'),
        Input('xaxis-range', 'data'),
        Input('cycle-select', 'value'),
        Input('calendar-range', 'data'),
    ],
    prevent_initial_call='initial_duplicate'
)
def update_graphs(slicer_clicks, xaxis_range, dropdown_cycle_value, cal_range):
    ctx = callback_context
    selected_id = 'global'; dropdown_value_to_set = None

    if ctx.triggered:
        tid = getattr(ctx, "triggered_id", None)
        if (isinstance(tid, str) and tid == 'cycle-select') or (ctx.triggered[0]['prop_id'] == 'cycle-select.value'):
            if dropdown_cycle_value is not None:
                selected_id = f'cycle-{int(dropdown_cycle_value)}'; dropdown_value_to_set = int(dropdown_cycle_value)
            else:
                selected_id = 'global'; dropdown_value_to_set = None
        else:
            if isinstance(tid, dict) and tid.get('type') == 'slicer':
                selected_id = tid['index']
            else:
                prop_id = ctx.triggered[0]['prop_id']
                if prop_id not in ('xaxis-range.data', 'calendar-range.data') and prop_id.startswith('{'):
                    try:
                        idx = json.loads(prop_id.split('.', 1)[0])
                        if idx and idx.get('type') == 'slicer':
                            selected_id = idx['index']
                    except Exception:
                        selected_id = 'global'

    df = record_df.copy()
    if not df.empty:
        df = _ensure_date_only(df)

    # 日历过滤
    cal_start = cal_end = None
    if cal_range and isinstance(cal_range, dict):
        try:
            cal_start = pd.to_datetime(cal_range.get('start')) if cal_range.get('start') else None
            cal_end = pd.to_datetime(cal_range.get('end')) if cal_range.get('end') else None
        except Exception:
            pass
    if not df.empty and cal_start is not None and cal_end is not None and 'Date' in df.columns:
        df = df[(df['Date'] >= cal_start) & (df['Date'] <= cal_end)]

    # slicer 过滤
    if selected_id != 'global' and not df.empty and 'Date' in df.columns:
        if selected_id.startswith('cycle-'):
            try:
                cycle_num = int(selected_id.split('-')[1])
                c = [c for c in cycle_info if c['cycle'] == cycle_num][0]
                s1, e1 = c['start'], c['end']
                if cal_start is not None and cal_end is not None:
                    s1 = max(s1, cal_start); e1 = min(e1, cal_end)
                df = df[(df['Date'] >= s1) & (df['Date'] <= e1)]
                if dropdown_value_to_set is None:
                    dropdown_value_to_set = cycle_num
            except Exception:
                df = record_df.copy(); dropdown_value_to_set = None
        elif selected_id.startswith('step-'):
            try:
                _, cycle_num, step_num = selected_id.split('-')
                cycle_num = int(cycle_num); step_num = int(step_num)
                s = [s for s in step_info if s['cycle'] == cycle_num and s['step'] == step_num][0]
                s1, e1 = s['start'], s['end']
                if cal_start is not None and cal_end is not None:
                    s1 = max(s1, cal_start); e1 = min(e1, cal_end)
                df = df[(df['Date'] >= s1) & (df['Date'] <= e1)]
                dropdown_value_to_set = cycle_num
            except Exception:
                df = record_df.copy(); dropdown_value_to_set = None
    else:
        if selected_id == 'global':
            dropdown_value_to_set = None

    # x 轴缩放过滤
    if xaxis_range and not df.empty and 'Date' in df.columns:
        try:
            xs = pd.to_datetime(xaxis_range['start']); xe = pd.to_datetime(xaxis_range['end'])
            df = df[(df['Date'] >= xs) & (df['Date'] <= xe)]
        except Exception:
            pass

    figures = [
        plot_current(df), plot_v0(df), plot_energy(df), plot_power(df),
        plot_soc(df), plot_voltage(df), plot_temp(df), plot_fault(df)
    ]
    slicer_components = create_slicer_bars(selected_id)
    return figures + [slicer_components, selected_id, dropdown_value_to_set]


# ------------------ 放大/收起 ------------------
@app.callback(
    Output('focused-graph', 'data', allow_duplicate=True),
    Input({'type': 'expand', 'index': ALL}, 'n_clicks'),
    State('focused-graph', 'data'),
    prevent_initial_call='initial_duplicate'
)
def set_focused_graph(expand_clicks, current_focus):
    ctx = callback_context
    if not ctx.triggered: return no_update
    tid = getattr(ctx, 'triggered_id', None)
    triggered_val = ctx.triggered[0]['value'] if ctx.triggered and ctx.triggered[0] else None
    if not triggered_val: return no_update
    if isinstance(tid, dict) and tid.get('type') == 'expand':
        clicked = tid.get('index')
        if clicked == '__close__' or current_focus == clicked:
            return None
        return clicked
    return no_update

@app.callback(
    [Output('focused-view', 'children'),
     Output('charts-grid-container', 'style')],
    [Input('focused-graph', 'data'),
     Input("current-graph", "figure"),
     Input("v0-graph", "figure"),
     Input("energy-graph", "figure"),
     Input("power-graph", "figure"),
     Input("soc-graph", "figure"),
     Input("volt-graph", "figure"),
     Input("temp-graph", "figure"),
     Input("fault-graph", "figure"),
     Input({'type': 'custom-graph', 'index': ALL}, 'figure'),
     Input('layout-applied', 'data')],
    prevent_initial_call=False
)
def render_focused_view(focused, fig_current, fig_v0, fig_energy, fig_power,
                        fig_soc, fig_volt, fig_temp, fig_fault, custom_figs, applied):
    if not focused:
        return (html.Div(), {'display': 'block'})

    mapping = {
        'current': fig_current, 'v0': fig_v0, 'energy': fig_energy, 'power': fig_power,
        'soc': fig_soc, 'volt': fig_volt, 'temp': fig_temp, 'fault': fig_fault
    }

    base_fig = mapping.get(focused); title = f"Focused: {focused}"

    if base_fig is None and isinstance(focused, str) and focused.startswith('custom-'):
        try:
            cid = int(focused.split('-', 1)[1]); pos = 0
            if applied:
                for i, it in enumerate(applied):
                    if int(it.get('id')) == cid:
                        pos = i; break
            if custom_figs and 0 <= pos < len(custom_figs):
                base_fig = custom_figs[pos]
        except Exception:
            base_fig = go.Figure()

    if base_fig is None:
        base_fig = go.Figure()

    fig_resized = go.Figure(base_fig)
    fig_resized.update_layout(autosize=True, width=None, height=520, margin=dict(l=60, r=30, t=50, b=40))

    big = dbc.Card([
        dbc.CardHeader(
            dbc.Row([
                dbc.Col(html.Strong(title), width="auto"),
                dbc.Col(
                    html.Button("Return", id={'type': 'expand', 'index': '__close__'},
                                n_clicks=0, title="Return",
                                style={'padding': '2px 8px','backgroundColor': '#f5f5f5',
                                       'border': '1px solid #999','borderRadius': '4px','cursor': 'pointer'}),
                    width="auto", className="ms-auto"
                )
            ], justify="between"), className="py-2"
        ),
        dbc.CardBody([
            dcc.Graph(id="focused-graph-figure", figure=fig_resized,
                      style={'width': '100%', 'height': '520px'},
                      config={'displaylogo': False, 'responsive': True})
        ], style={'padding': '6px'})
    ], style={'border': '1px solid #ddd'})

    return (big, {'display': 'none'})


# ------------------ 缩放同步 ------------------
@app.callback(
    Output('xaxis-range', 'data'),
    [Input("current-graph", "relayoutData"),
     Input("v0-graph", "relayoutData"),
     Input("energy-graph", "relayoutData"),
     Input("power-graph", "relayoutData"),
     Input("soc-graph", "relayoutData"),
     Input("volt-graph", "relayoutData"),
     Input("temp-graph", "relayoutData"),
     Input("fault-graph", "relayoutData"),
     Input({'type': 'custom-graph', 'index': ALL}, 'relayoutData')],
    prevent_initial_call=True
)
def sync_xaxis_range(*relayout_datas):
    ctx = callback_context
    if not ctx.triggered: return no_update
    relayout_data = ctx.triggered[0]['value']
    if relayout_data and 'xaxis.range[0]' in relayout_data:
        return {'start': relayout_data['xaxis.range[0]'], 'end': relayout_data['xaxis.range[1]']}
    return no_update


# ====================== Layout Setting（统一调度） ======================
def _record_cols_for_layout():
    """获取 record 列名（除 Date 外）"""
    if record_df is None or record_df.empty:
        return []
    cols = [str(c).strip() for c in record_df.columns if str(c).strip() != 'Date']
    return cols

@app.callback(
    [
        Output("layout-modal", "is_open"),
        Output("layout-staging", "data"),
        Output("layout-applied", "data"),
        Output("layout-id-seq", "data"),
    ],
    [
        Input("layout-btn", "n_clicks"),
        Input("layout-close", "n_clicks"),
        Input("layout-cancel", "n_clicks"),
        Input("layout-apply", "n_clicks"),
        Input("layout-add", "n_clicks"),
        Input('upload-layout', 'contents'),
        Input({'type': 'layout-del', 'index': ALL}, 'n_clicks'),
        Input({'type': 'layout-select', 'index': ALL}, 'value'),
    ],
    [
        State("layout-modal", "is_open"),
        State("layout-staging", "data"),
        State("layout-applied", "data"),
        State("layout-id-seq", "data"),
    ],
    prevent_initial_call=True
)
def layout_dispatcher(btn_open, btn_close, btn_cancel, btn_apply, btn_add, upload_contents,
                      del_clicks, select_values, is_open, staging, applied, id_seq):
    ctx = callback_context
    staging = staging or []; applied = applied or []; id_seq = int(id_seq or 0)
    if not ctx.triggered:
        return is_open, staging, applied, id_seq
    tid = ctx.triggered_id

    # 打开
    if tid == "layout-btn":
        new_staging = [dict(id=int(item['id']), columns=list(item.get('columns', []))) for item in applied]
        new_id_seq = max([int(item['id']) for item in new_staging], default=id_seq)
        return True, new_staging, applied, new_id_seq

    # 关闭（不应用）
    if tid in ("layout-close", "layout-cancel"):
        return False, staging, applied, id_seq

    # 应用
    if tid == "layout-apply":
        new_applied = [dict(id=int(item['id']), columns=list(item.get('columns', []))) for item in staging]
        new_id_seq = max([int(item['id']) for item in new_applied], default=id_seq)
        return False, staging, new_applied, new_id_seq

    # Add
    if tid == "layout-add":
        new_staging = list(staging)
        if len(new_staging) < 20:
            new_id = id_seq + 1
            new_staging.append({'id': new_id, 'columns': []})
            return is_open, new_staging, applied, new_id
        return is_open, staging, applied, id_seq

    # Load CSV
    if tid == "upload-layout":
        if not upload_contents:
            return is_open, staging, applied, id_seq
        try:
            content_type, content_string = upload_contents.split(',')
            decoded = base64.b64decode(content_string)
            df = pd.read_csv(io.BytesIO(decoded))
            new_list = []
            for _, row in df.iterrows():
                s_id = str(row.get('id')).strip() if pd.notna(row.get('id')) else ''
                if not s_id: continue
                cid = int(float(s_id))
                cols = str(row.get('columns')) if pd.notna(row.get('columns')) else ""
                selected = [c.strip() for c in cols.split(';') if c and c.strip()]
                new_list.append({'id': cid, 'columns': selected})
            new_list = new_list[:20]
            new_id_seq = max([int(x['id']) for x in new_list], default=id_seq)
            return is_open, new_list, applied, new_id_seq
        except Exception:
            return is_open, staging, applied, id_seq

    # 删除（pattern）
    if isinstance(tid, dict) and tid.get('type') == 'layout-del':
        del_id = int(tid.get('index'))
        new_list = [item for item in staging if int(item.get('id')) != del_id]
        return is_open, new_list, applied, id_seq

    # 多选变化（pattern）
    if isinstance(tid, dict) and tid.get('type') == 'layout-select':
        sel_id = int(tid.get('index'))
        value_list = ctx.triggered[0]['value'] or []
        new_list = []
        for item in staging:
            if int(item.get('id')) == sel_id:
                new_list.append({'id': sel_id, 'columns': value_list})
            else:
                new_list.append(item)
        return is_open, new_list, applied, id_seq

    return is_open, staging, applied, id_seq

# 渲染编辑区（卡片在上，Add 按钮在下）
@app.callback(
    Output("layout-editor", "children"),
    [Input("layout-staging", "data"), Input("layout-modal", "is_open")],
    prevent_initial_call=False
)
def render_layout_editor(staging, is_open):
    options = [{'label': c, 'value': c} for c in _record_cols_for_layout()]
    staging = staging or []; tiles = []
    counter = html.Div(f"Charts: {len(staging)} / 20", className="text-muted", style={'fontSize': '12px', 'marginBottom': '6px'})
    for item in staging:
        cid = int(item.get('id')); selected = item.get('columns', [])
        tiles.append(
            dbc.Card(
                dbc.CardBody([
                    html.Div(
                        [
                            html.Strong(f"Chart #{cid}"),
                            html.Button("×", id={'type': 'layout-del', 'index': cid}, n_clicks=0, title="删除该图",
                                        className="btn btn-link",
                                        style={'float': 'right', 'textDecoration': 'none', 'color': '#6c757d',
                                               'fontSize': '18px', 'lineHeight': '1'})
                        ],
                        className="mb-2"
                    ),
                    dcc.Dropdown(
                        id={'type': 'layout-select', 'index': cid},
                        options=options, value=selected, multi=True,
                        placeholder="Slect to print（1 or more）",
                        style={'minWidth': '280px'},
                        persistence=True, persistence_type='session'
                    )
                ]),
                className="mb-2",
                style={'border': '1px solid #e9ecef', 'borderRadius': '6px', 'boxShadow': '0 1px 2px rgba(0,0,0,0.03)'}
            )
        )
    plus_tile = html.Div(
        dbc.Button("+ Add Chart", id="layout-add", color="secondary", outline=True,
                   className="w-100", disabled=(len(staging) >= 20)),
        style={'border': '2px dashed #ced4da', 'borderRadius': '6px', 'padding': '16px',
               'textAlign': 'center', 'marginTop': '6px'}
    )
    return [counter, *tiles, plus_tile]

# 保存 CSV（保存 staging）
@app.callback(
    Output("download-layout", "data"),
    Input("save-layout-btn", "n_clicks"),
    State("layout-staging", "data"),
    prevent_initial_call=True
)
def save_layout(n_clicks, staging):
    if not n_clicks: return no_update
    staging = staging or []
    rows = [{'id': int(it.get('id')), 'columns': ';'.join(it.get('columns', []))} for it in staging] or [{'id': '', 'columns': ''}]
    df = pd.DataFrame(rows)
    return dcc.send_data_frame(df.to_csv, filename="battery_monkey_layout.csv", index=False)

# Apply 后重置筛选/缩放（避免旧状态过滤）—— 允许重复写
@app.callback(
    [
        Output('selected-slicer', 'data', allow_duplicate=True),
        Output('cycle-select', 'value', allow_duplicate=True),
        Output('xaxis-range', 'data', allow_duplicate=True),
        Output('date-range', 'start_date', allow_duplicate=True),
        Output('date-range', 'end_date', allow_duplicate=True),
        Output('calendar-range', 'data', allow_duplicate=True),
    ],
    Input('layout-apply', 'n_clicks'),
    prevent_initial_call=True
)
def reset_filters_after_apply(n_clicks):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    # 若不想清空日期，把后三个改成 no_update
    return 'global', None, {}, None, None, None

# 显示已应用状态
@app.callback(
    [
        Output("applied-count-badge", "children"),
        Output("applied-brief", "children")
    ],
    Input("layout-applied", "data"),
    prevent_initial_call=False
)
def show_applied_status(applied):
    items = applied or []; count = len(items)
    brief_list = []
    for i, it in enumerate(items[:3]):
        cid = it.get('id')
        cols = [str(c).strip() for c in it.get('columns', [])]
        brief_list.append(f"#{cid}: {', '.join(cols) if cols else 'No cols'}")
    more = f" (+{len(items)-3} more)" if len(items) > 3 else ""
    return str(count), (";  ".join(brief_list) + more) if items else "No applied charts"

# 自定义图渲染（统一用 Date 作为 x 轴）
@app.callback(
    Output("custom-charts-container", "children"),
    [Input("layout-applied", "data"),
     Input("calendar-range", "data"),
     Input("xaxis-range", "data"),
     Input("selected-slicer", "data")],
    prevent_initial_call=False
)
def render_custom_charts(applied, cal_range, xaxis_range, selected_id):
    if not applied:
        return []
    df = record_df.copy()
    if not df.empty:
        df = _ensure_date_only(df)

    # 1) 日历过滤
    cal_start = cal_end = None
    if cal_range and isinstance(cal_range, dict):
        try:
            cal_start = pd.to_datetime(cal_range.get('start')) if cal_range.get('start') else None
            cal_end = pd.to_datetime(cal_range.get('end')) if cal_range.get('end') else None
        except Exception:
            pass
    if not df.empty and cal_start is not None and cal_end is not None and 'Date' in df.columns:
        df = df[(df['Date'] >= cal_start) & (df['Date'] <= cal_end)]

    # 2) Slicer 过滤
    if selected_id and selected_id != 'global' and not df.empty and 'Date' in df.columns:
        try:
            if selected_id.startswith('cycle-'):
                cycle_num = int(selected_id.split('-')[1])
                c = [c for c in cycle_info if c['cycle'] == cycle_num][0]
                s1, e1 = c['start'], c['end']
                if cal_start is not None and cal_end is not None:
                    s1 = max(s1, cal_start); e1 = min(e1, cal_end)
                df = df[(df['Date'] >= s1) & (df['Date'] <= e1)]
            elif selected_id.startswith('step-'):
                _, cycle_num, step_num = selected_id.split('-')
                cycle_num = int(cycle_num); step_num = int(step_num)
                s = [s for s in step_info if s['cycle'] == cycle_num and s['step'] == step_num][0]
                s1, e1 = s['start'], s['end']
                if cal_start is not None and cal_end is not None:
                    s1 = max(s1, cal_start); e1 = min(e1, cal_end)
                df = df[(df['Date'] >= s1) & (df['Date'] <= e1)]
        except Exception:
            pass

    # 3) x 轴缩放过滤
    if xaxis_range and not df.empty and xaxis_range.get('start') and xaxis_range.get('end') and 'Date' in df.columns:
        try:
            xs = pd.to_datetime(xaxis_range['start']); xe = pd.to_datetime(xaxis_range['end'])
            df = df[(df['Date'] >= xs) & (df['Date'] <= xe)]
        except Exception:
            pass

    # 生成每个自定义图卡片（4 列栅格）
    cards = []
    for item in applied:
        cid = int(item.get('id'))
        cols = [str(c).strip() for c in (item.get('columns', []) or [])]

        fig = go.Figure()
        if not df.empty and cols and 'Date' in df.columns:
            for c in cols:
                if c in df.columns:
                    fig.add_trace(go.Scatter(x=df['Date'], y=df[c], mode='lines', name=c))

        fig.update_layout(
            template="plotly_white",
            font=dict(family="Segoe UI, Microsoft YaHei", size=12, color="#2c3e50"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=30, r=10, t=30, b=30), height=260,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0, font=dict(size=10)),
            uirevision=f'custom-{cid}',
            title=f"Custom #{cid}  —  {', '.join(cols) if cols else 'No columns selected'}"
        )

        cards.append(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div(
                                [
                                    html.Span(f"Custom #{cid}", className="text-muted", style={'fontWeight': 600}),
                                    html.Button(
                                        "⤢", id={'type': 'expand', 'index': f'custom-{cid}'},
                                        n_clicks=0, title="放大 / Focus",
                                        className="btn btn-sm btn-outline-secondary",
                                        style={'padding': '2px 6px', 'float': 'right'}
                                    )
                                ],
                                className="mb-2"
                            ),
                            dcc.Graph(
                                id={'type': 'custom-graph', 'index': cid},
                                figure=fig,
                                config={'displaylogo': False, 'responsive': True},
                                style={'height': '260px'}
                            )
                        ]
                    ),
                    className="shadow-sm border-0 h-100"
                ),
                md=3, className="mb-3"
            )
        )

    return dbc.Row(cards, className="gy-3")


# ------------------ 启动（本地/Render） ------------------
if __name__ == "__main__":
    # Render 会注入 PORT 环境变量；本地默认 8050
    port = int(os.environ.get("PORT", "8050"))
    app.run(host="0.0.0.0", port=port, debug=False)
