"""Query the local YAM recording catalog from the command line (the Refine step).

Talks to the catalog served by ``yam_rerun/server.py`` at
``rerun+http://localhost:51234``. The metadata stamped on each take (episode name,
task, curation tag) comes back as ``property:...`` columns, and any scores written
by ``tools/episode_metrics.py`` show up as ``property:metrics:...`` columns::

    python tools/query_dataset.py                                # list datasets
    python tools/query_dataset.py --dataset towels               # one row per episode
    python tools/query_dataset.py --dataset towels --tag "Good episode"
    python tools/query_dataset.py --dataset towels --full        # + metric columns
    python tools/query_dataset.py --dataset towels --episode episode_01 \
        --entity left_arm/position                               # series -> pandas

Ported from so100-hackathon ``tools/apps/query_dataset.py``; entity paths and the
default catalog port follow the YAM contract, and tyro/rich were swapped for
argparse + a plain table printer to keep the dependency set at rerun/datafusion/
numpy/pandas.
"""

from __future__ import annotations

import argparse

import pandas as pd

import _yam_catalog as yc
from datafusion import col, lit

import rerun as rr  # noqa: E402  (env prepared by _yam_catalog import)

SEGMENT_COLUMNS = {
    "rerun_segment_id": "segment_id",
    "property:RecordingInfo:name": "episode",
    "property:episode:task": "task",
    "property:episode:tag": "tag",
    "property:metrics:flag": "flag",
    "rerun_size_bytes": "size",
}

METRIC_COLUMN_PREFIX = "property:metrics:"


def list_datasets(client: rr.catalog.CatalogClient) -> None:
    names = sorted(client.dataset_names())
    if not names:
        print("no datasets yet -- record one first (yam_rerun collection hook)")
        return
    print(f"{len(names)} dataset(s) in the catalog:\n")
    for name in names:
        count = len(client.get_dataset(name=name).segment_ids())
        print(f"  {name:30s} {count} episode(s)")
    print("\ndetails: python tools/query_dataset.py --dataset <name>")


def show_segment_table(dataset: rr.catalog.DatasetEntry, tag: str | None, full: bool) -> pd.DataFrame:
    table = dataset.segment_table()
    if tag:
        # The tag filter runs inside DataFusion, on the server side of the reader.
        table = table.filter(col("property:episode:tag")[0] == lit(tag))
    df = table.to_pandas()
    if df.empty:
        print(f"no episodes{f' tagged {tag!r}' if tag else ''} in dataset '{dataset.name}'")
        return df

    columns = dict(SEGMENT_COLUMNS)
    if full:
        for name in df.columns:
            if name.startswith(METRIC_COLUMN_PREFIX) and name not in columns:
                columns[name] = name.removeprefix(METRIC_COLUMN_PREFIX)
    view = df[[column for column in columns if column in df.columns]].rename(columns=columns)
    for column in view.columns:
        if column not in ("segment_id", "size"):
            view[column] = view[column].map(yc.flatten)
    view = view.fillna("")
    if {"time:start", "time:end"} <= set(df.columns):
        seconds = (df["time:end"] - df["time:start"]).dt.total_seconds()
        view["duration"] = seconds.map(lambda s: f"{s:.1f}s")
    if "size" in view.columns:
        view["size"] = view["size"].map(yc.human_size)
    view = view.sort_values("episode").reset_index(drop=True)
    print(f"dataset '{dataset.name}'{f', tag {tag!r}' if tag else ''}: {len(view)} episode(s)")
    yc.print_table(view)
    return view


def show_entity_series(dataset: rr.catalog.DatasetEntry, *, episode: str | None, tag: str | None, entity: str) -> None:
    view = dataset
    if episode is not None:
        matches = [ep.segment_id for ep in yc.list_episodes(dataset, episode=episode)]
        if not matches:
            raise SystemExit(f"no episode named '{episode}' in dataset '{dataset.name}' (see --dataset {dataset.name} for the list)")
        view = view.filter_segments(matches)
    elif tag:
        view = view.filter_segments(dataset.segment_table().filter(col("property:episode:tag")[0] == lit(tag)))

    df = view.filter_contents([entity]).reader(index=yc.TIME_INDEX).to_pandas()
    data_columns = [column for column in df.columns if column not in ("rerun_segment_id", "log_time", "log_tick")]
    if len(data_columns) <= 1:  # only the index column came back
        entities = ", ".join(str(path) for path in dataset.schema().entity_paths())
        raise SystemExit(f"entity '{entity}' has no data here; entities in this dataset: {entities}")

    scope = f"episode '{episode}'" if episode else (f"episodes tagged {tag!r}" if tag else "all episodes")
    print(f"'{entity}' across {scope}: {len(df)} rows")
    yc.print_table(df[data_columns].head(10), max_cell=48)
    numeric = df[data_columns].select_dtypes("number")
    if not numeric.empty:
        print("summary:")
        yc.print_table(numeric.describe().rename_axis("stat").reset_index())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=None, help="catalog dataset to inspect; omit to list all datasets")
    parser.add_argument("--tag", default=None, help='only episodes with this curation tag (e.g. "Good episode")')
    parser.add_argument("--episode", default=None, help="zoom into one episode by name (as shown in the episode column)")
    parser.add_argument("--entity", default=None, help="entity path to pull as a series (e.g. left_arm/position)")
    parser.add_argument("--full", action="store_true", help="include all property:metrics:* score columns in the table")
    parser.add_argument("--catalog-port", type=int, default=yc.DEFAULT_CATALOG_PORT)
    args = parser.parse_args()

    client = yc.connect(args.catalog_port)
    if args.dataset is None:
        list_datasets(client)
        return
    dataset = client.get_dataset(name=args.dataset)
    if args.entity is not None:
        show_entity_series(dataset, episode=args.episode, tag=args.tag, entity=args.entity)
    else:
        show_segment_table(dataset, args.tag, args.full)


if __name__ == "__main__":
    try:
        main()
    except ConnectionError as error:
        raise SystemExit(f"cannot reach the catalog -- is yam_rerun/server.py running? ({error})") from None
