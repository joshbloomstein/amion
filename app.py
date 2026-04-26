#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import pandas as pd
import re
import requests

from datetime import datetime
from io import StringIO
from urllib.parse import quote
from shiny import App, reactive, render, ui


def generate_url(startdate, enddate, passkey):
    urlstem = f'https://www.amion.com/cgi-bin/ocs?Lo={passkey}&Rpt=625ctabs'
    y, m, d = startdate.strftime('%y'), startdate.month, startdate.day
    delta = (enddate - startdate).days
    return f'{urlstem}&Day={d}&Month={m}-{y}&Days={delta}'


def fetch_table(url):
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/plain",
        "Connection": "close",
    }

    try:
        r = requests.get(url, headers=headers, timeout=60)
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"REQUEST FAILED: {e}")

    text = r.text

    if "<html" in text.lower():
        raise RuntimeError(
            "Received HTML instead of table "
            "(invalid Access Code or blocked)"
        )

    if len(text.strip()) < 100:
        raise RuntimeError(
            "Response too short "
            "(likely blocked or bad Access Code)"
        )

    return StringIO(text)


def _make_exclude_regex():
    banned_terms = [
        'Conf', 'Didactic', 'Exam', 'Panel', 'Retreat', 'R1', 'R2', 'R3',
        'SOM Resc', 'Resc', 'ABIM', 'Board Prep',
        'Chief', 'Clinic', 'Holiday', 'Off', 'Immersion', 'Academic',
        'Vacation', 'Sick', 'Interview', 'PPC', 'Shadow', 'TBD', 'Jury',
        'ACGME', 'ACLS', 'BELL Outpatient', 'H Med', 'Immersion', 'PC',
        'RaTL', 'Panel Handoff', 'Risk', 'Bereavement', 'QI Project',
        'Graduated', 'Health Equity', 'H ER', 'H MICU',
        'Just-in-Time', 'Orientation', 'U ER',
        'Vac', 'nan', 'H Neuro', 'U Med', 'V Med',
        'V Night Med', 'V MICU', 'V Cards', 'U Cards A',
        'Elective', 'U HO', 'LWOP', 'H Geri', 'H CCU',
        'H Cards Consult', 'GIM', 'GME', 'Precept',
        'Stud Eve H', 'Swing', 'Primary Care'
    ]

    pattern = r'(?:' + r'|'.join(re.escape(t) for t in banned_terms) + r')'

    return re.compile(pattern, flags=re.IGNORECASE)


_EXCLUDE_RE = _make_exclude_regex()


def download_df(academicYear, passkey):

    if academicYear == 'AY23':
        startdate = datetime(2023, 6, 28)
        enddate = datetime(2024, 6, 30)

    elif academicYear == 'AY24':
        startdate = datetime(2024, 7, 1)
        enddate = datetime(2025, 6, 29)

    elif academicYear == 'AY25':
        startdate = datetime(2025, 6, 30)
        enddate = datetime(2026, 6, 29)

    elif academicYear == 'AY26':
        startdate = datetime(2026, 6, 30)
        enddate = datetime(2027, 6, 29)

    else:
        return pd.DataFrame([])

    url = generate_url(startdate, enddate, quote(passkey))
    file_like = fetch_table(url)

    try:
        text = file_like.getvalue()
        lines = text.splitlines()
        data_lines = [l for l in lines if "\t" in l and "," in l]

        if len(data_lines) == 0:
            raise RuntimeError("No valid data lines found")

        df_raw = pd.read_csv(
            StringIO("\n".join(data_lines)),
            sep="\t",
            header=None,
            engine="python"
        )

        cols = df_raw.shape[1]

        df = pd.DataFrame({
            'Name': df_raw.iloc[:, 0],
            'Assignment': df_raw.iloc[:, 3] if cols > 3 else '',
            'Date': df_raw.iloc[:, 6] if cols > 6 else '',
            'Start': df_raw.iloc[:, 7] if cols > 7 else '',
            'Stop': df_raw.iloc[:, 8] if cols > 8 else '',
            'Role': df_raw.iloc[:, 9] if cols > 9 else '',
            'Type': df_raw.iloc[:, 15] if cols > 15 else '',
            'Assgn': df_raw.iloc[:, 16] if cols > 16 else '',
        })

    except Exception:
        return pd.DataFrame([])

    df.columns = [
        'Name', 'Assignment', 'Date', 'Start',
        'Stop', 'Role', 'Type', 'Assgn'
    ]

    df = df[~df.Role.isnull()]
    df = df[df.Role != 'Services']
    df = df[df.Role.astype(str).str[-1] != '*']

    df['Name'] = (
        df['Name']
        .astype(str)
        .str.replace("'", '')
        .str.strip()
    )

    df['Assignment'] = (
        df['Assignment']
        .astype(str)
        .str.strip()
        .str.replace(
            r',\s*(am|pm)\s*$',
            '',
            regex=True,
            flags=re.IGNORECASE
        )
        .str.replace(r'\s+', ' ', regex=True)
    )

    df = df[df['Assignment'].notna()]
    df = df[df['Assignment'] != '']
    df = df[
        ~df['Assignment'].str.contains(
            _EXCLUDE_RE,
            na=False
        )
    ]

    return df


def build_master_rotations(df):
    return sorted(
        df['Assignment']
        .dropna()
        .unique()
        .tolist()
    )


def get_block_dates(year, block):

    blocks = {

        'AY25-26': {
            '1': ('2025-06-25', '2025-07-22'),
            '2': ('2025-07-23', '2025-08-19'),
            '3': ('2025-08-20', '2025-09-16'),
            '4': ('2025-09-17', '2025-10-14'),
            '5': ('2025-10-15', '2025-11-11'),
            '6': ('2025-11-12', '2025-12-09'),
            '7': ('2025-12-10', '2026-01-06'),
            '8': ('2026-01-07', '2026-02-03'),
            '9': ('2026-02-04', '2026-03-03'),
            '10': ('2026-03-04', '2026-03-31'),
            '11': ('2026-04-01', '2026-04-28'),
            '12': ('2026-04-29', '2026-05-26'),
            '13': ('2026-05-27', '2026-06-24'),
        },

        'AY26-27': {
            '1': ('2026-06-25', '2026-07-22'),
            '2': ('2026-07-23', '2026-08-19'),
            '3': ('2026-08-20', '2026-09-16'),
            '4': ('2026-09-17', '2026-10-14'),
            '5': ('2026-10-15', '2026-11-11'),
            '6': ('2026-11-12', '2026-12-09'),
            '7': ('2026-12-10', '2027-01-06'),
            '8': ('2027-01-07', '2027-02-03'),
            '9': ('2027-02-04', '2027-03-03'),
            '10': ('2027-03-04', '2027-03-31'),
            '11': ('2027-04-01', '2027-04-28'),
            '12': ('2027-04-29', '2027-05-26'),
            '13': ('2027-05-27', '2027-06-24'),
        }

    }

    return blocks[year][block]


def rotations_unfilled_in_block(df, master, year, block):

    start_date, end_date = get_block_dates(year, block)

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    df_block = df.loc[start:end]

    filled = set(df_block['Assignment'])

    return sorted(set(master) - filled)


app_ui = ui.page_fluid(

    ui.h3('Amion Rotation Openings Checker'),

    ui.input_text(
        'passkey',
        'Amion Access Code'
    ),

    ui.layout_sidebar(

        ui.sidebar(

            ui.input_select(
                'year',
                'Academic Year',
                [
                    'AY23-24',
                    'AY24-25',
                    'AY25-26',
                    'AY26-27'
                ],
                selected='AY26-27'
            ),

            ui.input_select(
                'block',
                'Block',
                [str(i) for i in range(1, 14)],
                selected='1'
            ),

            ui.input_action_button(
                'load',
                'Load / Refresh data'
            ),

            ui.input_action_button(
                'check',
                'Check block'
            )

        ),

        ui.div(
            ui.output_text_verbatim('status'),
            ui.hr(),
            ui.h4('All assignments (count)'),
            ui.output_text('master_count'),
            ui.h4(
                'Assignments that may have openings'
            ),
            ui.output_table('unfilled_table'),
        )

    )

)


def server(input, output, session):

    df_state = reactive.Value(
        pd.DataFrame([])
    )

    master_state = reactive.Value([])

    status_state = reactive.Value(
        'Ready'
    )

    unfilled_state = reactive.Value(
        pd.DataFrame([])
    )

    @reactive.Effect
    @reactive.event(input.load)
    def load_data():

        pk = (
            input.passkey() or ''
        ).strip()

        if pk == '':
            status_state.set(
                'No Access Code entered.'
            )
            return

        try:
            status_state.set(
                'Loading...'
            )

            selected_year = input.year()

            year_map = {
                'AY23-24': 'AY23',
                'AY24-25': 'AY24',
                'AY25-26': 'AY25',
                'AY26-27': 'AY26'
            }

            amion_year = year_map[
                selected_year
            ]

            df = download_df(
                amion_year,
                pk
            )

            if df.empty:
                status_state.set(
                    'No data returned.'
                )
                return

            df['Date'] = (
                df['Date']
                .astype(str)
                .str.strip()
            )

            df['Date_dt'] = pd.to_datetime(
                df['Date'],
                format='%m-%d-%y',
                errors='coerce'
            )

            df = df[
                df['Date_dt'].notna()
            ]

            df = df.sort_values(
                'Date_dt'
            )

            df = df.set_index(
                'Date_dt'
            )

            master = build_master_rotations(
                df
            )

            df_state.set(df)
            master_state.set(master)

            status_state.set(
                f'{selected_year} '
                f'loaded rows = {len(df)}, '
                f'all assignments = '
                f'{len(master)}'
            )

        except Exception as e:
            status_state.set(
                f'Load failed: {e}'
            )

    @reactive.Effect
    @reactive.event(input.check)
    def check_block():

        df = df_state.get()
        master = master_state.get()

        if df.empty:
            status_state.set(
                'Load data first.'
            )
            return

        try:
            selected_year = input.year()
            block = input.block()

            result = rotations_unfilled_in_block(
                df,
                master,
                selected_year,
                block
            )

            unfilled_state.set(
                pd.DataFrame({
                    'Assignment': result
                })
            )

            status_state.set(
                f'{selected_year} '
                f'Block {block}: '
                f'{len(result)} openings found'
            )

        except Exception as e:
            status_state.set(
                f'Check failed: {e}'
            )

    @output
    @render.text
    def status():
        return status_state.get()

    @output
    @render.text
    def master_count():
        return str(
            len(
                master_state.get()
            )
        )

    @output
    @render.table
    def unfilled_table():
        return unfilled_state.get()


app = App(app_ui, server)