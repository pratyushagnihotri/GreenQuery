import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full", auto_download=["html"])


@app.cell(hide_code=True)
def _():
    import base64
    import gzip
    import json
    from pathlib import Path

    import marimo as mo
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _embedded_data = {
        "movie": (
            "H4sIAAAAAAAC/+2dXW8bV5KG7/MrBF+PNac+Tn0sMBczwCyQRRY72Q32ZjAgGImJtSOLDkXZyQ72v28VZUmM2E3Zao8RkyVYtsw+"
            "pCjx8O23nq6Pv351cvKP+Dw5eXG+eL188S8nL14v314sXvzu9sbXy/PF5ebWxXr+8vJy/nr++2/y75d0Ci/tTy+/vrper27O1nd3"
            "eHM5v7parGYffcfVzVUuj39mDe5u/Olmsfolb/4WZm+W1xfri7eL2WoRT/Dd9dYd1xevF7Prxdny6vw6Vnc6VSfQjtjYvL9fuIjn"
            "9eMvs7+/exVr2mlr2NRMnFSbCnrH9wvPljj7cbNGoQNBk67AyF3uHunnxdnN+mJ5Nbu4enOznq2Xf19c5XeG1pruLFrerB+vevTb"
            "evQw8bzx0Yqdx0C5+7nWy/X8cusIdeX3hy6XZ3FotYjf4/U6D9qjh906RHc//fzs1WL26mJz490TPb9YLc7Ws5/ms+/n67NXs+uL"
            "/13E4auby8v3K95eXOfP+vSKu52xdSRf9Kv568X1m/nZozs9/sZny3i5t57YTzfzy4t17pHbbRw3Xa/n65t88i8Wb+eXN/P14vz9"
            "XjnZbMfVxVkefBMPfPuM5uv4PVzfXK5nlxevL9YPi1eL9c3qanE+Wy3fbTbW/ZGz5WrzvHYO3D9qvhKn7f72H1fxxM9nseXXrzb7"
            "/HYTz2IvnF7Hoy2uF5ttfLU++cMfTv7yH//19Xdf//efX2zu/n93L9qrX64v8gVdvlms5uvlKr/1X+++wze/fu5/WS3/Z3G2dcN3"
            "i5/X387/9eJyvVi9/Oabf9/8/49/mp/Fvjl/+dTb9OXZKt4yt8/nb79+Q10vb1ably1+ufPrm1X8vn5YLV/Prt6+vpxt3sGzdxdX"
            "58t3ee/Nz/JFKA7uKM7sh+Vqtp7H72tG4+rD7ZTQqJEBOTGhjOkPdGdVBjcRxea7+tPZpCEhE8O9PI2rj9jT4oNPao/DE9rjisPS"
            "w65tTHm8lGeC8tCY8uCg8rRTefRRQvTFChHd68/1fKM071+fcQkKs3LaejNzdmzEPqJAEEusNwWQWL7jf8JGhedhYVZHIPWn7Q87"
            "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"
            "wjJA1VoVzimc8xkt0Fd/++r/Acj/wD6UbAEA"
        ),
        "rotowire": (
            "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"
            "Xrx9xQAA"
        ),
        "medical": (
            "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"
        ),
    }
    _data_root = Path(__file__).resolve().parent.parent / "data"
    MODEL_LABELS = {
        "Qwen/Qwen2.5-7B-Instruct": "Qwen 2.5 7B",
        "Qwen/Qwen2.5-14B-Instruct": "Qwen 2.5 14B",
        "meta-llama/Llama-3.1-8B-Instruct": "Llama 3.1 8B",
    }
    DATASETS = {
        "movie": {"label": "Movie", "source": _data_root / "movie_query_metrics.json", "expected_queries": 4, "context": "Movie-review queries, including filters, a relational join, and a ratio estimate."},
        "rotowire": {"label": "RotoWire", "source": _data_root / "rotowire_query_metrics.json", "expected_queries": 2, "context": "Basketball-report queries that join records and repeatedly extract facts from text."},
        "medical": {"label": "Medical", "source": _data_root / "medical_query_metrics.json", "expected_queries": 4, "context": "Patient-record queries, including text and image-QA workloads."},
    }
    QUERY_GUIDES = {
        "movie": {"Q1": "Find positive reviews", "Q2": "Find positive reviews for Taken 3", "Q3": "Find review pairs with the same sentiment", "Q4": "Estimate Taken 3's positive-review ratio"},
        "rotowire": {"R1": "Extract each player's points and assists", "R2": "Extract player points where the home team won"},
        "medical": {"Q1": "Find patients with allergies", "Q3": "Find cancer patients with family and lung conditions", "Q4": "Average the age of patients with acne", "Q9": "Find sick patients with skin X-ray records"},
    }

    def load_dataset(dataset):
        if dataset not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset}")
        spec = DATASETS[dataset]
        try:
            rows = json.loads(spec["source"].read_text())
        except OSError:
            rows = json.loads(gzip.decompress(base64.b64decode(_embedded_data[dataset])))
        frame = pd.DataFrame(rows).copy()
        models = set(frame["model"])
        expected = len(models) * 10 * spec["expected_queries"]
        if len(frame) != expected or frame[["model", "run", "query"]].duplicated().any():
            raise ValueError(f"{spec['label']} must have {expected} unique model/run/query measurements.")
        if len(models) < 2 or not models <= set(MODEL_LABELS) or (frame["cache_hits"] != 0).any():
            raise ValueError(f"{spec['label']} is not a complete supported-model cold-cache comparison.")
        if (frame["planner_model"] != frame["model"]).any() or not (frame["energy_source"] == "measured_from_nvml_query_window").all():
            raise ValueError(f"{spec['label']} does not have the expected matched NVML measurements.")
        frame["model_label"] = frame["model"].map(MODEL_LABELS)
        frame["query_id"] = frame["query"].str.split("_", n=1).str[0]
        frame["query_description"] = frame["query"].str.split("_", n=1).str[1].str.replace("_", " ")
        frame["energy_mwh"] = frame["energy_kwh"] * 1_000_000
        frame["run_number"] = frame["run"].str[-2:].astype(int)
        return frame

    return DATASETS, MODEL_LABELS, QUERY_GUIDES, go, load_dataset, make_subplots, mo


@app.cell(hide_code=True)
def _(DATASETS, mo):
    dataset_picker = mo.ui.radio(
        {spec["label"]: key for key, spec in DATASETS.items()},
        value="Movie",
        label="Dataset",
    )
    mo.sidebar(
        [
            mo.md("## Explore a dataset\n\nSwitch the whole dashboard at any time."),
            dataset_picker,
        ],
        footer=mo.md("Matched planner · 10 cold-cache repetitions per model"),
        width="270px",
    )
    return (dataset_picker,)


@app.cell(hide_code=True)
def _(DATASETS, MODEL_LABELS, dataset_picker, load_dataset, mo):
    selected_key = dataset_picker.value
    selected = DATASETS[selected_key]
    records = load_dataset(selected_key)
    query_count = records["query"].nunique()
    mo.vstack([
        mo.md("# Qwen vs Llama: Different Query Structures\n\n### Energy, runtime, tokens, and task-aware quality"),
        mo.callout(
            mo.md("**Fully local experiment:** no API calls; the same local model performs planning and execution; planner and answer caches are disabled; GPU energy is measured from NVML."),
            kind="info",
        ),
        mo.md(f"## {selected['label']}\n\n{selected['context']}"),
        mo.hstack([
            mo.stat(label="Queries", value=str(query_count), caption="Same query set for both models."),
            mo.stat(label="Matched repetitions", value="10", caption="One Qwen and one Llama run per query ID."),
            mo.stat(label="Validated records", value=f"{len(records)}", caption="Complete model × run × query grid."),
            mo.stat(label="Models", value="2 local", caption="Qwen plans Qwen; Llama plans Llama."),
        ], widths="equal", gap=1),
    ])
    return records, selected_key


@app.cell(hide_code=True)
def _(QUERY_GUIDES, go, make_subplots, mo, records, selected_key):
    _metrics = [
        ("energy_mwh", "GPU energy", "mWh"),
        ("runtime_seconds", "Runtime", "seconds"),
        ("co2_g", "Recorded CO₂", "g"),
        ("total_tokens", "Total tokens", "tokens"),
    ]
    _models = [("Qwen 2.5 7B", "#7C3AED", -0.22), ("Llama 3.1 8B", "#0F766E", 0.22)]
    _queries = list(records.sort_values("query_id")["query_id"].drop_duplicates())
    _formats = {
        "energy_mwh": lambda value: f"{value / 1_000:.1f}k" if value >= 10_000 else f"{value:,.0f}",
        "runtime_seconds": lambda value: f"{value:,.0f}",
        "co2_g": lambda value: f"{value:.2f}",
        "total_tokens": lambda value: f"{value / 1_000_000:.2f}M" if value >= 1_000_000 else f"{value / 1_000:.1f}k",
    }
    _titles = [f"<b>{title}</b><br><sup>{unit}</sup>" for _, title, unit in _metrics]
    _figure = make_subplots(rows=2, cols=2, subplot_titles=_titles, horizontal_spacing=0.10, vertical_spacing=0.18)
    for _metric_index, (column, title, unit) in enumerate(_metrics):
        _row, _column = divmod(_metric_index, 2)
        for model, color, offset in _models:
            subset = records[records["model_label"] == model]
            _x, _y, _hover = [], [], []
            _medians_x, _medians_y = [], []
            for position, query in enumerate(_queries):
                points = subset[subset["query_id"] == query].sort_values("run_number")
                jitter = (points["run_number"] - 5.5) * 0.026
                _x.extend(position + offset + jitter)
                _y.extend(points[column])
                _hover.extend(points[["run", "query_description", "local_requests"]].to_numpy())
                _medians_x.append(position + offset)
                _medians_y.append(points[column].median())
            _figure.add_scatter(
                x=_x, y=_y, mode="markers", marker=dict(color=color, size=7, opacity=0.58),
                name=model, legendgroup=model, showlegend=_metric_index == 0, customdata=_hover,
                hovertemplate=(f"{model}<br>%{{customdata[0]}} · %{{customdata[1]}}<br>"
                               f"{title}: %{{y:,.3f}} {unit}<br>Local requests: %{{customdata[2]}}<extra></extra>"),
                row=_row + 1, col=_column + 1,
            )
            _figure.add_scatter(
                x=_medians_x, y=_medians_y, mode="markers",
                marker=dict(color=color, size=20, symbol="line-ew", line=dict(color="white", width=2)),
                text=[_formats[column](value) for value in _medians_y],
                textposition="top center" if model == "Qwen 2.5 7B" else "bottom center",
                textfont=dict(color=color, size=11),
                legendgroup=model, showlegend=False, hoverinfo="skip",
                row=_row + 1, col=_column + 1,
            )
        _figure.update_xaxes(tickmode="array", tickvals=list(range(len(_queries))), ticktext=_queries, range=[-0.5, len(_queries) - 0.5], row=_row + 1, col=_column + 1)
        _figure.update_yaxes(rangemode="tozero", gridcolor="#E2E8F0", row=_row + 1, col=_column + 1)
    _figure.update_layout(
        height=760, margin=dict(l=55, r=25, t=90, b=55), hovermode="closest",
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="center", x=0.5),
    )
    _energy_medians = records.groupby("query_id")["energy_mwh"].median().sort_values(ascending=False)
    _hotspot = _energy_medians.index[0]
    _comparison = records.pivot(index=["run", "query_id"], columns="model_label", values="energy_mwh")
    _qwen_lower = int((_comparison["Qwen 2.5 7B"] < _comparison["Llama 3.1 8B"]).sum())
    _scope = {"movie": "Movie and Medical values are in the low-thousands of mWh; RotoWire is much larger because it repeatedly extracts from 728 reports.", "rotowire": "RotoWire is on a larger resource scale than Movie and Medical because each query repeatedly extracts facts from 728 reports.", "medical": "Medical shares Movie’s low-thousands-of-mWh scale, while RotoWire is much larger because it repeatedly extracts from 728 reports."}[selected_key]
    _guide_text = " · ".join(f"**{query}** — {description}" for query, description in QUERY_GUIDES[selected_key].items())
    _energy_runtime_correlation = records["energy_mwh"].corr(records["runtime_seconds"])
    _power = (records["energy_mwh"] * 3.6 / records["runtime_seconds"])
    _power_by_query = _power.groupby(records["query_id"]).median().reindex(_queries)
    _power_text = " · ".join(f"**{query}** {power:.0f} W" for query, power in _power_by_query.items())
    _totals = records.groupby("model_label")[["energy_mwh", "co2_g", "runtime_seconds", "total_tokens"]].sum()

    def _total_row(metric, label, formatter):
        qwen, llama = _totals.loc["Qwen 2.5 7B", metric], _totals.loc["Llama 3.1 8B", metric]
        higher, lower, winner = (qwen, llama, "Qwen") if qwen > llama else (llama, qwen, "Llama")
        return {"Metric": label, "Qwen": formatter(qwen), "Llama": formatter(llama), "Higher total": f"{winner} (+{(higher / lower - 1) * 100:.1f}%)"}

    _overall = [
        _total_row("energy_mwh", "Measured GPU energy", lambda value: f"{value / 1_000:.1f} Wh"),
        _total_row("co2_g", "Recorded CO₂", lambda value: f"{value:.2f} g"),
        _total_row("runtime_seconds", "End-to-end runtime", lambda value: f"{value / 60:.1f} min"),
        _total_row("total_tokens", "Total tokens", lambda value: f"{value / 1_000_000:.2f}M" if value >= 1_000_000 else f"{value / 1_000:.1f}k"),
    ]
    mo.vstack([
        mo.md("---\n\n## 2. Measured resources\n\nEvery small dot is one cold-cache run. The thick horizontal tick is the median of ten runs. Hover any dot to inspect its run ID and local-request count."),
        mo.md(f"**{selected_key.title()} query guide:** {_guide_text}"),
        mo.md("### Overall footprint across all matched runs"),
        mo.ui.table(_overall, pagination=False, selection=None),
        mo.callout(mo.md("**Carbon note:** recorded CO₂ is calculated from the NVML-measured GPU energy with the same configured emissions factor for both models. Therefore energy and CO₂ have the same ranking; tokens are shown as workload, not as a carbon estimate."), kind="info"),
        mo.ui.plotly(_figure, config={"displayModeBar": False}),
        mo.callout(mo.md(f"**{_hotspot} is the energy hotspot** for the selected dataset. Qwen is lower-energy in **{_qwen_lower}/ {len(_comparison)}** matched query runs.\n\n**Energy ↔ runtime readout:** correlation **{_energy_runtime_correlation:.3f}** · median GPU power: {_power_text}.\n\n{_scope}"), kind="info"),
    ])
    return


@app.cell(hide_code=True)
def _(QUERY_GUIDES, go, make_subplots, mo, records, selected_key):
    _metrics = [("energy_mwh", "Median energy", "Wh", 1 / 1_000), ("runtime_seconds", "Median runtime", "seconds", 1)]
    _medians = records.groupby(["query_id", "model_label"])[[metric for metric, _, _, _ in _metrics]].median().unstack("model_label")
    _figure = make_subplots(rows=1, cols=2, subplot_titles=["<b>Median energy: Qwen ← → Llama</b>", "<b>Median runtime: Qwen ← → Llama</b>"], horizontal_spacing=0.22)
    _largest = {}
    for _index, (_metric, _title, _unit, _scale) in enumerate(_metrics, start=1):
        _rows = []
        for _query, _description in QUERY_GUIDES[selected_key].items():
            _qwen = _medians.loc[_query, (_metric, "Qwen 2.5 7B")] * _scale
            _llama = _medians.loc[_query, (_metric, "Llama 3.1 8B")] * _scale
            if _qwen < _llama:
                _gap, _winner = (_llama - _qwen) / _llama * 100, "Qwen"
            elif _llama < _qwen:
                _gap, _winner = (_qwen - _llama) / _qwen * 100, "Llama"
            else:
                _gap, _winner = 0, "Tie"
            _rows.append({"label": f"{_query} — {_description}", "gap": _gap, "winner": _winner, "qwen": _qwen, "llama": _llama})
        _rows.sort(key=lambda _entry: abs(_entry["gap"]), reverse=True)
        _largest[_metric] = _rows[0]
        _labels = [_entry["label"] for _entry in _rows]
        _figure.add_bar(
            x=[-_entry["qwen"] for _entry in _rows],
            y=_labels,
            orientation="h",
            marker_color="#7C3AED",
            name="Qwen 2.5 7B",
            legendgroup="Qwen 2.5 7B",
            showlegend=_index == 1,
            text=[f"{_entry['qwen']:.2f}" if _unit == "Wh" else f"{_entry['qwen']:,.0f}" for _entry in _rows],
            textposition="outside",
            cliponaxis=False,
            customdata=[[_entry["qwen"], _entry["llama"], _entry["winner"], _entry["gap"]] for _entry in _rows],
            hovertemplate=(f"%{{y}}<br>{_title}: Qwen %{{customdata[0]:,.3f}} {_unit} · Llama %{{customdata[1]:,.3f}} {_unit}<br>"
                           "%{customdata[2]} %{customdata[3]:.1f}% lower<extra></extra>"),
            row=1,
            col=_index,
        )
        _figure.add_bar(
            x=[_entry["llama"] for _entry in _rows],
            y=_labels,
            orientation="h",
            marker_color="#0F766E",
            name="Llama 3.1 8B",
            legendgroup="Llama 3.1 8B",
            showlegend=_index == 1,
            text=[f"{_entry['llama']:.2f}" if _unit == "Wh" else f"{_entry['llama']:,.0f}" for _entry in _rows],
            textposition="outside",
            cliponaxis=False,
            hoverinfo="skip",
            row=1,
            col=_index,
        )
        _limit = max(max(_entry["qwen"], _entry["llama"]) for _entry in _rows) * 1.35
        _ticks = [-_limit, -_limit / 2, 0, _limit / 2, _limit]
        _tick_text = [f"{abs(value):.1f}" if _unit == "Wh" else f"{abs(value):,.0f}" for value in _ticks]
        _figure.update_xaxes(range=[-_limit, _limit], tickmode="array", tickvals=_ticks, ticktext=_tick_text, zeroline=True, zerolinewidth=2, zerolinecolor="#475569", title=f"{_unit} · Qwen ←   → Llama", row=1, col=_index)
        _figure.update_yaxes(categoryorder="array", categoryarray=list(reversed(_labels)), row=1, col=_index)
    _figure.update_layout(height=max(360, 82 * len(QUERY_GUIDES[selected_key])), margin=dict(l=20, r=90, t=90, b=60), plot_bgcolor="white", paper_bgcolor="white", barmode="overlay", legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5))
    _energy = _largest["energy_mwh"]
    _runtime = _largest["runtime_seconds"]
    mo.vstack([
        mo.md("---\n\n## 3. Who had the lower footprint?\n\nThis mirrored tornado shows both **ten-run medians** for every query: Qwen extends left in purple; Llama extends right in teal. Bar-end labels are the actual medians; hover shows the percentage advantage."),
        mo.ui.plotly(_figure, config={"displayModeBar": False}),
        mo.callout(mo.md(f"**Largest energy gap:** {_energy['label']} — {_energy['winner']} is **{abs(_energy['gap']):.1f}% lower**. **Largest runtime gap:** {_runtime['label']} — {_runtime['winner']} is **{abs(_runtime['gap']):.1f}% lower**. CO₂ is not repeated here because it uses the same energy ranking."), kind="info"),
    ])
    return


@app.cell(hide_code=True)
def _(go, mo, records):
    def _operator_label(operators):
        _text = " ".join(operators)
        if "ImageQa" in _text:
            return "image-QA"
        if "ExtractAndQa" in _text:
            return "text extraction + QA"
        if "TextQaExtract" in _text and "TextQaFilter" in _text:
            return "text extraction + filter"
        if "TextQaExtract" in _text:
            return "text extraction"
        if "TextQaFilter" in _text:
            return "text QA filter"
        return "database-only plan"

    _order = list(records.groupby("query_id")["total_tokens"].median().sort_values(ascending=False).index)
    _styles = {
        "Qwen 2.5 7B": ("#C4B5FD", "#7C3AED"),
        "Llama 3.1 8B": ("#99D7D0", "#0F766E"),
    }
    _rows = []
    for _query in _order:
        _query_records = records[records["query_id"] == _query]
        _description = _query_records["query_description"].iat[0]
        _operator = _operator_label(_query_records["physical_operators"].iat[0])
        for _model in _styles:
            _model_records = _query_records[_query_records["model_label"] == _model]
            _planning = (_model_records["planner_input_tokens"] + _model_records["planner_output_tokens"]).median()
            _execution = (_model_records["execution_input_tokens"] + _model_records["execution_output_tokens"]).median()
            _rows.append({
                "label": f"{_query} — {_description}<br><sup>{_operator} · {_model}</sup>",
                "query": _query,
                "model": _model,
                "operator": _operator,
                "planning": _planning,
                "execution": _execution,
                "total": _planning + _execution,
            })

    _figure = go.Figure()
    for _model, (_planning_color, _execution_color) in _styles.items():
        _model_rows = [_row for _row in _rows if _row["model"] == _model]
        _custom = [[_row["query"], _row["operator"], _row["planning"], _row["execution"], _row["total"]] for _row in _model_rows]
        _figure.add_bar(
            y=[_row["label"] for _row in _model_rows],
            x=[_row["planning"] for _row in _model_rows],
            orientation="h",
            name=f"{_model} · planning",
            marker_color=_planning_color,
            customdata=_custom,
            hovertemplate=("%{customdata[0]} · " + _model + "<br>"
                           "Planner tokens: %{customdata[2]:,.0f}<br>"
                           "Execution tokens: %{customdata[3]:,.0f}<br>"
                           "Total: %{customdata[4]:,.0f}<br>"
                           "Primary local step: %{customdata[1]}<extra></extra>"),
        )
        _figure.add_bar(
            y=[_row["label"] for _row in _model_rows],
            x=[_row["execution"] for _row in _model_rows],
            orientation="h",
            name=f"{_model} · execution",
            marker_color=_execution_color,
            text=[f"{_row['total'] / 1_000_000:.2f}M" if _row["total"] >= 1_000_000 else f"{_row['total'] / 1_000:.1f}k" for _row in _model_rows],
            textposition="outside",
            cliponaxis=False,
            customdata=_custom,
            hovertemplate=("%{customdata[0]} · " + _model + "<br>"
                           "Planner tokens: %{customdata[2]:,.0f}<br>"
                           "Execution tokens: %{customdata[3]:,.0f}<br>"
                           "Total: %{customdata[4]:,.0f}<br>"
                           "Primary local step: %{customdata[1]}<extra></extra>"),
        )
    _figure.update_layout(
        barmode="stack",
        height=max(440, 96 * len(_rows)),
        margin=dict(l=415, r=100, t=85, b=60),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="center", x=0.5),
        xaxis=dict(title="Median tokens across 10 runs", tickformat="~s", gridcolor="#E2E8F0", rangemode="tozero"),
        yaxis=dict(categoryorder="array", categoryarray=list(reversed([_row["label"] for _row in _rows]))),
    )
    _hotspot = max(_rows, key=lambda _row: _row["total"])
    _planning_share = _hotspot["planning"] / _hotspot["total"] * 100
    _totals = {model: sum(_row["total"] for _row in _rows if _row["model"] == model) for model in _styles}
    _higher_model = max(_totals, key=_totals.get)
    _phase_columns = {
        "phase_energy_valid",
        "phase_energy_provenance",
        "phase_energy_source",
        "planner_energy_kwh",
        "execution_energy_kwh",
        "unattributed_energy_kwh",
    }
    _show_phase_energy = (
        _phase_columns.issubset(records.columns)
        and records["phase_energy_valid"].fillna(False).all()
        and records["phase_energy_provenance"].eq("nvml_non_overlapping_local_inference_windows_v1").all()
        and records["phase_energy_source"].eq("measured_from_nvml_phase_windows").all()
        and records[list(_phase_columns - {"phase_energy_valid", "phase_energy_provenance", "phase_energy_source"})].notna().all().all()
    )
    _phase_body = []
    if _show_phase_energy:
        _phase_rows = []
        for _query in _order:
            _query_records = records[records["query_id"] == _query]
            for _model in _styles:
                _model_records = _query_records[_query_records["model_label"] == _model]
                _phase_rows.append({
                    "label": f"{_query} — {_model}",
                    "planning": _model_records["planner_energy_kwh"].median() * 1_000_000,
                    "execution": _model_records["execution_energy_kwh"].median() * 1_000_000,
                    "unattributed": _model_records["unattributed_energy_kwh"].median() * 1_000_000,
                })
        _phase_figure = go.Figure()
        for _phase, _color in (("planning", "#7C3AED"), ("execution", "#0F766E"), ("unattributed", "#94A3B8")):
            _phase_figure.add_bar(
                y=[row["label"] for row in _phase_rows],
                x=[row[_phase] for row in _phase_rows],
                orientation="h",
                name=_phase.capitalize(),
                marker_color=_color,
                hovertemplate="%{y}<br>" + _phase.capitalize() + ": %{x:,.1f} mWh<extra></extra>",
            )
        _phase_figure.update_layout(
            barmode="stack",
            height=max(440, 70 * len(_phase_rows)),
            margin=dict(l=240, r=60, t=65, b=60),
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="center", x=0.5),
            xaxis=dict(title="Median NVML energy across 10 runs (mWh)", gridcolor="#E2E8F0", rangemode="tozero"),
            yaxis=dict(categoryorder="array", categoryarray=list(reversed([row["label"] for row in _phase_rows]))),
        )
        _phase_body = [
            mo.md("### Planner versus execution energy\n\nThis split is shown only when every run has valid, non-overlapping timestamp-window provenance. Unattributed energy is measured query-window energy outside those local inference windows."),
            mo.ui.plotly(_phase_figure, config={"displayModeBar": False}),
        ]
    mo.vstack([
        mo.md("---\n\n## 4. What drove model work?\n\nEach row is one model’s **ten-run median** for a query. Light colour is planning—choosing a plan; dark colour is execution—doing the selected local-model work. Rows are ordered from the largest total-token workload downward."),
        mo.ui.plotly(_figure, config={"displayModeBar": False}),
        mo.callout(mo.md(f"**Largest token workload:** {_hotspot['query']} for {_hotspot['model']} at **{_hotspot['total']:,.0f} tokens**; **{100 - _planning_share:.1f}%** is execution work through the labelled {_hotspot['operator']} step. **{_higher_model}** has the larger summed median token workload in this dataset. Tokens explain workload; they are not a direct carbon measurement."), kind="info"),
        *_phase_body,
    ])
    return


@app.cell(hide_code=True)
def _(QUERY_GUIDES, go, make_subplots, mo, records, selected_key):
    def _quality_values(query, model, key):
        _values = [item.get(key) for item in records[(records["query_id"] == query) & (records["model_label"] == model)]["quality"] if isinstance(item, dict) and item.get(key) is not None]
        _values.sort()
        return (_values[(len(_values) - 1) // 2] + _values[len(_values) // 2]) / 2 if _values else None

    if selected_key == "movie":
        _models = [("Qwen 2.5 7B", "#7C3AED"), ("Llama 3.1 8B", "#0F766E")]
        _precision_queries = ["Q1", "Q2"]
        _figure = make_subplots(
            rows=1,
            cols=3,
            column_widths=[0.42, 0.25, 0.33],
            subplot_titles=[
                "<b>Q1–Q2: precision</b><br><sup>higher is better</sup>",
                "<b>Q3: pair F1</b><br><sup>higher is better</sup>",
                "<b>Q4: ratio error</b><br><sup>lower is better</sup>",
            ],
            horizontal_spacing=0.13,
        )
        for _model, _color in _models:
            _precision = [_quality_values(query, _model, "precision") for query in _precision_queries]
            _details = []
            for _query in _precision_queries:
                _item = records[(records["query_id"] == _query) & (records["model_label"] == _model)]["quality"].iat[0]
                _details.append([_item["correct_rows"], _item["returned_rows"]])
            _figure.add_bar(
                x=_precision_queries,
                y=[value * 100 for value in _precision],
                name=_model,
                marker_color=_color,
                text=[f"{value * 100:.1f}%" for value in _precision],
                textposition="outside",
                cliponaxis=False,
                customdata=_details,
                hovertemplate="%{x} · " + _model + "<br>Precision: %{y:.1f}%<br>Correct / returned: %{customdata[0]} / %{customdata[1]}<extra></extra>",
                row=1,
                col=1,
            )
        _q3_f1 = [_quality_values("Q3", model, "f1") for model, _ in _models]
        _figure.add_bar(
            x=[model for model, _ in _models],
            y=[value * 100 for value in _q3_f1],
            name="Pair F1",
            marker_color=[color for _, color in _models],
            text=[f"{value * 100:.1f}%" for value in _q3_f1],
            textposition="outside",
            cliponaxis=False,
            showlegend=False,
            hovertemplate="%{x}<br>Median pair F1: %{y:.1f}%<extra></extra>",
            row=1,
            col=2,
        )
        _figure.add_scatter(
            x=["Qwen 2.5 7B"], y=[0], mode="markers+text", text=["0.0%"], textposition="top center",
            marker=dict(color="#7C3AED", size=10, symbol="circle"),
            showlegend=False,
            hovertemplate="Qwen 2.5 7B<br>Median pair F1: 0.0%<br>Qwen returned zero review pairs in every run.<extra></extra>",
            row=1, col=2,
        )
        _errors = [_quality_values("Q4", model, "absolute_error") * 100 for model, _ in _models]
        _figure.add_bar(
            x=[model for model, _ in _models],
            y=_errors,
            marker_color=[color for _, color in _models],
            text=[f"{value:.1f} pp" for value in _errors],
            textposition="outside",
            cliponaxis=False,
            showlegend=False,
            hovertemplate="%{x}<br>True ratio: 11.67%<br>Absolute error: %{y:.1f} percentage points<extra></extra>",
            row=1,
            col=3,
        )
        _figure.update_yaxes(range=[0, 112], ticksuffix="%", gridcolor="#E2E8F0", row=1, col=1)
        _figure.update_yaxes(range=[0, 100], ticksuffix="%", gridcolor="#E2E8F0", row=1, col=2)
        _figure.update_yaxes(rangemode="tozero", title="percentage points", gridcolor="#E2E8F0", row=1, col=3)
        _figure.update_xaxes(categoryorder="array", categoryarray=["Qwen 2.5 7B", "Llama 3.1 8B"], row=1, col=2)
        _figure.update_layout(
            barmode="group", height=470, margin=dict(l=50, r=45, t=120, b=60),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="center", x=0.5),
        )
        _body = [
            mo.md("---\n\n## 5. Task-aware quality\n\nMovie has evaluated quality evidence for every query. Q3 is a genuine zero-output result for Qwen: it returned zero same-sentiment review pairs in every repetition, giving precision, recall, and F1 of 0.0%."),
            mo.ui.plotly(_figure, config={"displayModeBar": False}),
            mo.callout(mo.md(f"**Q1** is a 100% precision tie. **Q2:** Llama returns 2 correct reviews from 3 (**66.7%**) versus Qwen’s 1 from 3 (**33.3%**). **Q3:** Qwen has F1 **{_q3_f1[0] * 100:.1f}%** because it returned zero pairs; Llama’s median F1 is **{_q3_f1[1] * 100:.1f}%**. **Q4:** Qwen is closer to the 11.67% true ratio—**{_errors[0]:.1f}** versus **{_errors[1]:.1f} percentage points** of error."), kind="info"),
        ]
    elif selected_key == "medical":
        _models = [("Qwen 2.5 7B", "#7C3AED"), ("Llama 3.1 8B", "#0F766E")]
        _f1_queries = [query for query in QUERY_GUIDES[selected_key] if _quality_values(query, "Qwen 2.5 7B", "f1") is not None]
        _f1_labels = [f"{query}<br><sup>{QUERY_GUIDES[selected_key][query]}</sup>" for query in _f1_queries]
        _figure = make_subplots(
            rows=1,
            cols=2,
            column_widths=[0.70, 0.30],
            subplot_titles=["<b>Patient-set retrieval: F1 score</b><br><sup>higher is better</sup>", "<b>Q1: precision versus recall</b><br><sup>why the F1 scores differ</sup>"],
            horizontal_spacing=0.14,
        )
        for _model, _color in _models:
            _f1 = [_quality_values(query, _model, "f1") for query in _f1_queries]
            _details = []
            for _query in _f1_queries:
                _item = records[(records["query_id"] == _query) & (records["model_label"] == _model)]["quality"].iat[0]
                _details.append([_item["precision"] * 100, _item["recall"] * 100, _item["correct_rows"], _item["returned_rows"], _item["expected_rows"]])
            _figure.add_bar(
                x=_f1_labels,
                y=[value * 100 for value in _f1],
                name=_model,
                marker_color=_color,
                text=[f"{value * 100:.1f}%" for value in _f1],
                textposition="outside",
                cliponaxis=False,
                customdata=_details,
                hovertemplate=("%{x}<br>F1: %{y:.1f}%<br>Precision: %{customdata[0]:.1f}%<br>"
                               "Recall: %{customdata[1]:.1f}%<br>Correct / returned / expected: %{customdata[2]} / %{customdata[3]} / %{customdata[4]}<extra></extra>"),
                row=1,
                col=1,
            )
        for _model, _color in _models:
            _item = records[(records["query_id"] == "Q1") & (records["model_label"] == _model)]["quality"].iat[0]
            _figure.add_bar(
                x=["Precision", "Recall"],
                y=[_item["precision"] * 100, _item["recall"] * 100],
                marker_color=_color,
                text=[f"{_item['precision'] * 100:.1f}%", f"{_item['recall'] * 100:.1f}%"],
                textposition="outside",
                cliponaxis=False,
                showlegend=False,
                customdata=[[_item["correct_rows"], _item["returned_rows"], _item["expected_rows"]]] * 2,
                hovertemplate=("Q1 · " + _model + "<br>%{x}: %{y:.1f}%<br>"
                               "Correct / returned / expected: %{customdata[0]} / %{customdata[1]} / %{customdata[2]}<extra></extra>"),
                row=1,
                col=2,
            )
        _figure.update_yaxes(range=[0, 112], ticksuffix="%", gridcolor="#E2E8F0", row=1, col=1)
        _figure.update_yaxes(range=[0, 112], ticksuffix="%", gridcolor="#E2E8F0", row=1, col=2)
        _figure.update_layout(
            barmode="group",
            height=480,
            margin=dict(l=55, r=45, t=90, b=85),
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="center", x=0.5),
        )
        _f1_reading = " · ".join(
            f"**{query}:** Qwen {_quality_values(query, 'Qwen 2.5 7B', 'f1') * 100:.1f}% vs Llama {_quality_values(query, 'Llama 3.1 8B', 'f1') * 100:.1f}%"
            for query in _f1_queries
        )
        _body = [
            mo.md("---\n\n## 5. Task-aware quality\n\nMedical has measured ground truth. The F1 chart compares patient-set retrieval tasks; the second chart explains the only meaningful F1 difference. Q4’s numeric age estimate is omitted here because both models produced the same 1.25-year error."),
            mo.ui.plotly(_figure, config={"displayModeBar": False}),
            mo.callout(mo.md(f"All ten repetitions produced the same measured quality values. {_f1_reading}. **Q1 reading:** Qwen is precise—2 correct patients from 2 returned—but finds only 2 of 5 expected patients. Llama finds 4 of 5, but returns 13 patients, so 9 are incorrect. F1 rewards the balance, giving Qwen the higher score."), kind="info"),
        ]
    else:
        _status = []
        for _query, _description in QUERY_GUIDES[selected_key].items():
            _items = list(records[records["query_id"] == _query]["quality"])
            if all(item is None for item in _items):
                _label, _detail = "Not evaluated", "No quality metric is stored in this final matched-planner input."
            elif any(isinstance(item, dict) and item.get("filter_ground_truth") for item in _items):
                _label, _detail = "Not evaluated", "Ground truth for the query condition is not available."
            else:
                _label, _detail = "No numeric score", "The source records zero checked fields, so accuracy is unavailable."
            _status.append({"query": _query, "description": _description, "label": _label, "detail": _detail})
        _evidence = [
            {"Query": f"{item['query']} — {item['description']}", "Qwen 2.5 7B": item["label"], "Llama 3.1 8B": item["label"], "Why no score?": item["detail"]}
            for item in _status
        ]
        _body = [
            mo.md("---\n\n## 5. Task-aware quality\n\nRotoWire has no usable numeric quality score. A chart would imply a measurement that does not exist, so this panel shows the exact evidence state instead."),
            mo.ui.table(_evidence, pagination=False, selection=None),
            mo.callout(mo.md("**R1:** both models have zero checked fields, so numeric accuracy is unavailable. **R2:** the shipped ground truth also lacks home-team-winner labels. Resource comparisons remain valid, but this dataset cannot support an answer-quality comparison."), kind="warn"),
        ]
    mo.vstack(_body)
    return


if __name__ == "__main__":
    app.run()
