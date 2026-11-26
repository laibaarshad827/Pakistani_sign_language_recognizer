
import pandas as pd

METADATA = {"sign", "session", "take", "frame"}


def analyze_recordings(csv_path):
    df = pd.read_csv(csv_path)

    feature_columns = [
        column for column in df.columns
        if column not in METADATA
    ]

    coordinate_columns = [
        column for column in feature_columns
        if "_a" not in column
    ]

    angle_columns = [
        column for column in feature_columns
        if "_a" in column
    ]

    if not coordinate_columns or not angle_columns:
        raise ValueError("Expected coordinate and finger-angle features.")

    # Scale features so coordinates and angles are comparable.
    scales = df[feature_columns].std().replace(0, 1).fillna(1)

    rows = []

    for (sign, session, take), group in df.groupby(
        ["sign", "session", "take"]
    ):
        group = group.sort_values("frame")
        values = group[feature_columns].astype(float)

        if len(values) < 2:
            continue

        normalized = (values - df[feature_columns].mean()) / scales
        changes = normalized.diff().iloc[1:]

        coordinate_change = changes[coordinate_columns].abs().mean().mean()
        angle_change = changes[angle_columns].abs().mean().mean()

        coordinate_variation = normalized[coordinate_columns].std().mean()
        angle_variation = normalized[angle_columns].std().mean()

        rows.append({
            "sign": sign,
            "session": session,
            "take": take,
            "frames": len(group),
            "coordinate_frame_change": coordinate_change,
            "angle_frame_change": angle_change,
            "coordinate_variation": coordinate_variation,
            "angle_variation": angle_variation
        })

    recording_metrics = pd.DataFrame(rows)

    if recording_metrics.empty:
        raise ValueError("No recordings with at least two frames were found.")

    sign_summary = recording_metrics.groupby("sign").agg(
        recordings=("take", "count"),
        avg_coordinate_change=("coordinate_frame_change", "mean"),
        avg_angle_change=("angle_frame_change", "mean"),
        avg_coordinate_variation=("coordinate_variation", "mean"),
        avg_angle_variation=("angle_variation", "mean")
    ).reset_index()

    return recording_metrics, sign_summary


if __name__ == "__main__":
    recordings, summary = analyze_recordings(
        "data/processed/features.csv"
    )

    print("\nTEMPORAL ANALYSIS")
    print("=" * 55)
    print("Recordings analyzed:", len(recordings))
    print("\nOverall averages:")
    print(recordings.select_dtypes(include="number").mean())

    print("\nSigns with highest average temporal variation:")
    print(
        summary.sort_values(
            "avg_angle_variation", ascending=False
        ).head(10).to_string(index=False)
    )

    recordings.to_csv(
        "reports/phase4/temporal_recordings.csv", index=False
    )
    summary.to_csv(
        "reports/phase4/temporal_sign_summary.csv", index=False
    )

    print("\nResults saved in reports/phase4/")
