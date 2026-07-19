"""Score, auto-tag, and compare YAM episodes straight from the Rerun catalog.

The curation loop this enables: record -> ``score`` a dataset -> review the flagged
episodes in the viewer -> retag -> export only the good ones. All numbers come out
of catalog queries (``reader(index="time", fill_latest_at=True)`` over the four
7-D joint streams); the scores go back INTO the catalog as a ``metrics`` property
layer, so they show up as ``property:metrics:*`` columns in every later query
(``tools/query_dataset.py --full``) and can drive DataFusion filters.

Metrics per episode (q = 14-D position, g = 14-D goal, both in radians; "arm"
means the 12 revolute joints, grippers excluded unless said otherwise):

- ``duration_s``       last - first timestamp on the ``time`` timeline
- ``frames``           complete frames (every stream has a value)
- ``path_length``      sum_t ||q_arm[t+1] - q_arm[t]||_2           [rad]
- ``jerk_rms``         RMS of ||d3 q_arm / dt3||_2 (np.gradient x3, non-uniform dt)
                       [rad/s^3] -- high = jerky teleop or oscillating policy
- ``idle_frac``        fraction of steps with ||dq_arm||_2 < eps   -- dead time
- ``gripper_toggles``  open/close transitions summed over both grippers
                       (binarized at each gripper's observed midpoint; a gripper
                       whose range is < min-range never toggles)
- ``track_rms``        RMS over frames & arm joints of (g - q)     [rad]
- ``track_max``        max |g - q| over frames & arm joints        [rad]

Outliers: per-dataset robust z-score (median / MAD, z = 0.6745*(x-med)/MAD) on
path_length, jerk_rms, idle_frac, track_rms and duration_s; any |z| above the
threshold suggests the tag "Needs review". ``--apply-tags`` writes that tag back
through the shared ``edits`` layer (same file convention as the server's retag).

Usage::

    python tools/episode_metrics.py score   --dataset towels
    python tools/episode_metrics.py score   --dataset towels --apply-tags
    python tools/episode_metrics.py compare --demo towels --rollout molmoact2_eval

``compare`` resamples EVERY episode of both datasets onto the same fixed-rate
grid (``using_index_values`` + ``fill_latest_at``), computes the same metrics on
the aligned frames, and prints one demo-vs-rollout divergence table -- where the
policy is slower, jerkier, or tracks worse than the human demonstrations.
"""

from __future__ import annotations

import argparse
import dataclasses

import numpy as np
import pandas as pd

import _yam_catalog as yc

MAD_Z = 0.6745  # standard consistency constant: z = MAD_Z * (x - median) / MAD
SCORED_METRICS = ("path_length", "jerk_rms", "idle_frac", "track_rms", "duration_s")


# --- Metric math ----------------------------------------------------------------------


def compute_metrics(t: np.ndarray, q: np.ndarray, g: np.ndarray, *, idle_eps: float) -> dict[str, float]:
    """All per-episode scores from aligned (T,) seconds / (T,14) position / (T,14) goal."""
    arm_q = q[:, yc.ARM_DIMS]
    arm_g = g[:, yc.ARM_DIMS]
    steps = np.linalg.norm(np.diff(arm_q, axis=0), axis=1)  # ||dq|| per step

    if len(t) >= 4:
        d1 = np.gradient(arm_q, t, axis=0)
        d2 = np.gradient(d1, t, axis=0)
        d3 = np.gradient(d2, t, axis=0)
        jerk_rms = float(np.sqrt(np.mean(np.sum(d3**2, axis=1))))
    else:
        jerk_rms = float("nan")

    toggles = 0
    for dim in yc.GRIPPER_DIMS:
        channel = q[:, dim]
        low, high = float(channel.min()), float(channel.max())
        if high - low < 0.01:  # gripper never actually moved
            continue
        closed = channel > (low + high) / 2.0
        toggles += int(np.count_nonzero(np.diff(closed)))

    error = arm_g - arm_q
    return {
        "duration_s": round(float(t[-1] - t[0]), 3),
        "frames": int(len(t)),
        "path_length": round(float(steps.sum()), 4),
        "jerk_rms": round(jerk_rms, 3),
        "idle_frac": round(float(np.mean(steps < idle_eps)) if len(steps) else 1.0, 4),
        "gripper_toggles": toggles,
        "track_rms": round(float(np.sqrt(np.mean(error**2))), 5),
        "track_max": round(float(np.abs(error).max()), 5),
    }


def robust_flags(scores: pd.DataFrame, *, threshold: float) -> tuple[pd.Series, pd.Series]:
    """(flag, reasons) per episode: any scored metric whose median/MAD z-score exceeds
    the threshold marks the episode "Needs review". MAD == 0 (all-identical values)
    disables that metric rather than dividing by zero."""
    reasons = pd.Series([[] for _ in range(len(scores))], index=scores.index, dtype=object)
    for metric in SCORED_METRICS:
        values = scores[metric].astype(float)
        if values.isna().any():
            continue
        median = values.median()
        deviations = (values - median).abs()
        # Robust scale: MAD when it is informative; otherwise (a majority of identical
        # values, common in tiny datasets) fall back to the mean absolute deviation.
        # Both scaled to be sigma-consistent under normality.
        if deviations.median() > 1e-12:
            scale = deviations.median() / MAD_Z
        elif deviations.mean() > 1e-12:
            scale = deviations.mean() / 0.7979
        else:
            continue
        z = (values - median) / scale
        for index in scores.index[z.abs() > threshold]:
            reasons[index].append(f"{metric} z={z[index]:+.1f}")
    flags = reasons.map(lambda r: "Needs review" if r else "")
    return flags, reasons.map(", ".join)


# --- Episode loading ------------------------------------------------------------------


@dataclasses.dataclass
class Loaded:
    episode: yc.Episode
    t: np.ndarray
    q: np.ndarray
    g: np.ndarray


def load_episode(dataset, episode: yc.Episode, *, hz: float | None = None) -> Loaded | None:
    """One aligned frame table for the 4 joint streams (single reader round-trip).

    With ``hz`` set the reader resamples onto a fixed-rate grid spanning the episode
    (``using_index_values`` + ``fill_latest_at``) instead of returning logged ticks --
    that puts every episode of every dataset on directly comparable time axes.
    """
    index_values = None
    if hz is not None:
        if episode.start is None or episode.end is None:
            return None
        start = pd.Timestamp(episode.start).value
        end = pd.Timestamp(episode.end).value
        step = int(1e9 / hz)
        if end <= start:
            return None
        index_values = np.arange(start, end + 1, step).astype("datetime64[ns]")
    df = yc.episode_frames(dataset, episode.segment_id, list(yc.JOINT_ENTITIES), index_values=index_values)
    if len(df) < 2:
        return None
    return Loaded(
        episode=episode,
        t=yc.times_seconds(df),
        q=yc.stack14(df, *yc.POSITION_ENTITIES),
        g=yc.stack14(df, *yc.GOAL_ENTITIES),
    )


def score_dataset(dataset, episodes: list[yc.Episode], *, idle_eps: float, hz: float | None = None) -> pd.DataFrame:
    rows = []
    for episode in episodes:
        loaded = load_episode(dataset, episode, hz=hz)
        if loaded is None:
            print(f"  {episode.name}: <2 complete frames, skipped")
            continue
        rows.append({"episode": episode.name, "segment_id": episode.segment_id, "tag": episode.tag, "task": episode.task}
                    | compute_metrics(loaded.t, loaded.q, loaded.g, idle_eps=idle_eps))
    return pd.DataFrame(rows)


# --- score subcommand -----------------------------------------------------------------


def cmd_score(args: argparse.Namespace) -> None:
    client = yc.connect(args.catalog_port)
    dataset = client.get_dataset(name=args.dataset)
    episodes = yc.list_episodes(dataset, episode=args.episode)
    if not episodes:
        raise SystemExit(f"no episodes in dataset '{args.dataset}'")

    scores = score_dataset(dataset, episodes, idle_eps=args.idle_eps)
    if scores.empty:
        raise SystemExit("no scorable episodes")
    scores["flag"], scores["reasons"] = robust_flags(scores, threshold=args.z_threshold)

    display = scores.drop(columns=["segment_id", "task"])
    print(f"dataset '{args.dataset}': {len(scores)} episode(s) scored")
    yc.print_table(display)

    flagged = scores[scores["flag"] != ""]
    if not flagged.empty:
        print(f"\n{len(flagged)} episode(s) suggested 'Needs review': " + ", ".join(flagged["episode"]))

    if args.no_write:
        print("\n--no-write: scores NOT written back to the catalog")
        return

    # Write the scoreboard back as the `metrics` layer (own layer + own property name,
    # so it never collides with the human-curated `edits` layer chunks).
    paths = []
    for _, row in scores.iterrows():
        payload = {metric: row[metric] for metric in
                   ("duration_s", "frames", "path_length", "jerk_rms", "idle_frac", "gripper_toggles", "track_rms", "track_max")}
        payload |= {"flag": row["flag"], "reasons": row["reasons"]}
        path = yc.metrics_path(args.dataset, row["episode"])
        yc.write_episode_metrics(path, recording_id=row["segment_id"], scores=payload)
        paths.append(path)
    yc.register_layer(client, args.dataset, "metrics", paths)
    print(f"\nscores written back: {len(paths)} metrics file(s) under {yc.metrics_path(args.dataset, 'x').parent}")
    print("  -> visible as property:metrics:* columns (try: python tools/query_dataset.py --dataset "
          f"{args.dataset} --full)")
    print("  (the server's startup rescan re-applies metrics/ files, so scores survive restarts)")

    if not flagged.empty and args.apply_tags:
        edit_paths = []
        for _, row in flagged.iterrows():
            episode = next(ep for ep in episodes if ep.segment_id == row["segment_id"])
            path = yc.edits_path(args.dataset, episode.name)
            yc.write_episode_edits(path, recording_id=episode.segment_id, dataset=episode.dataset,
                                   task=episode.task, tag="Needs review")
            edit_paths.append(path)
        yc.register_layer(client, args.dataset, "edits", edit_paths)
        print(f"applied tag 'Needs review' to {len(edit_paths)} episode(s) via the edits layer")
    elif not flagged.empty:
        print("(--apply-tags would retag them 'Needs review' via the edits layer)")


# --- compare subcommand ---------------------------------------------------------------

COMPARE_METRICS = ("duration_s", "path_length", "jerk_rms", "idle_frac", "gripper_toggles", "track_rms", "track_max")


def cmd_compare(args: argparse.Namespace) -> None:
    client = yc.connect(args.catalog_port)
    sides = {}
    for label, name, tag in (("demo", args.demo, args.demo_tag), ("rollout", args.rollout, args.rollout_tag)):
        dataset = client.get_dataset(name=name)
        episodes = yc.list_episodes(dataset, tag=tag or None)
        if not episodes:
            raise SystemExit(f"no episodes{f' tagged {tag!r}' if tag else ''} in {label} dataset '{name}'")
        print(f"{label}: dataset '{name}', {len(episodes)} episode(s), resampled at {args.hz:g} Hz")
        scores = score_dataset(dataset, episodes, idle_eps=args.idle_eps, hz=args.hz)
        if scores.empty:
            raise SystemExit(f"no scorable episodes in {label} dataset '{name}'")
        sides[label] = scores

    rows = []
    for metric in COMPARE_METRICS:
        demo = sides["demo"][metric].astype(float)
        rollout = sides["rollout"][metric].astype(float)
        delta = rollout.mean() - demo.mean()
        rows.append({
            "metric": metric,
            "demo mean±std": f"{demo.mean():.4g} ± {demo.std(ddof=0):.2g}",
            "rollout mean±std": f"{rollout.mean():.4g} ± {rollout.std(ddof=0):.2g}",
            "delta": f"{delta:+.4g}",
            "delta %": f"{100 * delta / demo.mean():+.0f}%" if abs(demo.mean()) > 1e-12 else "n/a",
        })
    print(f"\ndemo ('{args.demo}') vs rollout ('{args.rollout}') on a shared {args.hz:g} Hz grid:")
    yc.print_table(pd.DataFrame(rows))
    print("\nreading guide: rollout jerk_rms / track_rms well above demo = policy oscillates or "
          "lags its own goals; idle_frac above demo = policy stalls; path_length above demo = detours.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    score = sub.add_parser("score", help="score every episode of a dataset, write scores back as a metrics layer")
    score.add_argument("--dataset", required=True)
    score.add_argument("--episode", default=None, help="score a single episode by name")
    score.add_argument("--idle-eps", type=float, default=0.005, help="||dq_arm|| below this counts a step as idle [rad]")
    score.add_argument("--z-threshold", type=float, default=3.5, help="|robust z| above this flags 'Needs review'")
    score.add_argument("--apply-tags", action="store_true", help="retag flagged episodes 'Needs review' via the edits layer")
    score.add_argument("--no-write", action="store_true", help="print the scoreboard only; do not touch the catalog")
    score.add_argument("--catalog-port", type=int, default=yc.DEFAULT_CATALOG_PORT)
    score.set_defaults(func=cmd_score)

    compare = sub.add_parser("compare", help="demo-vs-rollout metric divergence on a shared resampled grid")
    compare.add_argument("--demo", required=True, help="teleop/demonstration dataset name")
    compare.add_argument("--rollout", required=True, help="policy rollout dataset name (e.g. molmoact2_eval)")
    compare.add_argument("--demo-tag", default="", help="restrict demo episodes to this tag")
    compare.add_argument("--rollout-tag", default="", help="restrict rollout episodes to this tag")
    compare.add_argument("--hz", type=float, default=10.0, help="shared resampling rate for both datasets")
    compare.add_argument("--idle-eps", type=float, default=0.005)
    compare.add_argument("--catalog-port", type=int, default=yc.DEFAULT_CATALOG_PORT)
    compare.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except ConnectionError as error:
        raise SystemExit(f"cannot reach the catalog -- is yam_rerun/server.py running? ({error})") from None
