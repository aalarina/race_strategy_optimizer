import pandas as pd


PROCESSED_DATA_PATH = (
    "data/processed/laps_with_quality_flags.parquet"
)


# ============================================================
# Loading
# ============================================================

def load_processed_laps():
    """
    Load processed lap data.
    """

    df = pd.read_parquet(
        PROCESSED_DATA_PATH
    )

    return df


# ============================================================
# Missing TyreLife blocks
# ============================================================

def find_missing_tyre_life_blocks(df):
    """
    Find consecutive blocks of missing TyreLife values
    for each driver, race, and stint.
    """

    df = df.copy()

    df["TyreLife"] = pd.to_numeric(
        df["TyreLife"],
        errors="coerce",
    )

    df = df.sort_values(
        [
            "Year",
            "GrandPrix",
            "Driver",
            "LapNumber",
        ]
    )

    results = []

    group_columns = [
        "Year",
        "GrandPrix",
        "Driver",
        "Stint",
    ]

    for group_keys, group in df.groupby(
        group_columns,
        dropna=False,
    ):

        group = group.sort_values(
            "LapNumber"
        ).copy()

        missing = group["TyreLife"].isna()

        if not missing.any():
            continue

        # Identify consecutive missing blocks.
        block_id = (
            missing.ne(
                missing.shift()
            )
            .cumsum()
        )

        for _, block in group[
            missing
        ].groupby(block_id[missing]):

            results.append(
                {
                    "Year": group_keys[0],
                    "GrandPrix": group_keys[1],
                    "Driver": group_keys[2],
                    "Stint": group_keys[3],

                    "start_lap": block[
                        "LapNumber"
                    ].min(),

                    "end_lap": block[
                        "LapNumber"
                    ].max(),

                    "n_missing": len(block),

                    "compound": block[
                        "Compound"
                    ].dropna().iloc[0]
                    if block["Compound"].notna().any()
                    else None,
                }
            )

    return pd.DataFrame(results)


# ============================================================
# Inspect missing blocks
# ============================================================

def inspect_missing_blocks(
    df,
    n=30,
):
    """
    Display missing TyreLife blocks together with
    the observations immediately before and after them.
    """

    blocks = find_missing_tyre_life_blocks(df)

    if blocks.empty:
        print(
            "No missing TyreLife blocks found."
        )
        return blocks

    print(
        f"Total missing blocks: {len(blocks)}"
    )

    print(
        "\nLargest missing blocks:"
    )

    print(
        blocks
        .sort_values(
            "n_missing",
            ascending=False,
        )
        .head(n)
        .to_string(index=False)
    )

    return blocks


# ============================================================
# Inspect one missing block in context
# ============================================================

def inspect_block_context(
    df,
    block,
):
    """
    Show laps around one missing TyreLife block.

    The block must contain:
        Year
        GrandPrix
        Driver
        Stint
        start_lap
        end_lap
    """

    mask = (
        (df["Year"] == block["Year"])
        & (df["GrandPrix"] == block["GrandPrix"])
        & (df["Driver"] == block["Driver"])
        & (df["Stint"] == block["Stint"])
    )

    group = (
        df.loc[mask]
        .sort_values("LapNumber")
        .copy()
    )

    start_lap = block["start_lap"]
    end_lap = block["end_lap"]

    context = group[
        group["LapNumber"].between(
            start_lap - 3,
            end_lap + 3,
        )
    ][
        [
            "Year",
            "GrandPrix",
            "Driver",
            "LapNumber",
            "Stint",
            "Compound",
            "TyreLife",
            "PitInTime",
            "PitOutTime",
            "TrackStatus",
            "LapTimeSeconds",
        ]
    ]

    print(
        context.to_string(index=False)
    )

    return context


# ============================================================
# Find potentially reconstructable blocks
# ============================================================

def find_reconstructable_blocks(df):
    """
    Find missing TyreLife blocks that may be reconstructed
    using observations before and after the missing block.

    The search is performed across the whole race for each driver,
    rather than being restricted to a single Stint.
    """

    df = df.copy()

    df["TyreLife"] = pd.to_numeric(
        df["TyreLife"],
        errors="coerce",
    )

    df = df.sort_values(
        [
            "Year",
            "GrandPrix",
            "Driver",
            "LapNumber",
        ]
    )

    blocks = find_missing_tyre_life_blocks(df)

    candidates = []

    # --------------------------------------------------------
    # Analyze each missing block
    # --------------------------------------------------------

    for _, block in blocks.iterrows():

        # IMPORTANT:
        # Search the entire race for this driver.
        # Do NOT restrict the search to the same Stint.
        mask = (
            (df["Year"] == block["Year"])
            & (
                df["GrandPrix"]
                == block["GrandPrix"]
            )
            & (
                df["Driver"]
                == block["Driver"]
            )
        )

        race_driver = (
            df.loc[mask]
            .sort_values("LapNumber")
            .copy()
        )

        # ----------------------------------------------------
        # Find the last known TyreLife before the block
        # ----------------------------------------------------

        before = race_driver[
            race_driver["LapNumber"]
            < block["start_lap"]
        ]

        before = before[
            before["TyreLife"].notna()
        ]

        # ----------------------------------------------------
        # Find the first known TyreLife after the block
        # ----------------------------------------------------

        after = race_driver[
            race_driver["LapNumber"]
            > block["end_lap"]
        ]

        after = after[
            after["TyreLife"].notna()
        ]

        if before.empty or after.empty:
            continue

        previous = before.iloc[-1]
        following = after.iloc[0]

        # ----------------------------------------------------
        # Calculate the gap
        # ----------------------------------------------------

        lap_gap = (
            following["LapNumber"]
            - previous["LapNumber"]
        )

        tyre_gap = (
            following["TyreLife"]
            - previous["TyreLife"]
        )

        # ----------------------------------------------------
        # Check compound transition
        # ----------------------------------------------------

        previous_compound = (
            previous["Compound"]
        )

        next_compound = (
            following["Compound"]
        )

        compound_changed = (
            pd.notna(previous_compound)
            and pd.notna(next_compound)
            and previous_compound
            != next_compound
        )

        # ----------------------------------------------------
        # Check whether a pit stop occurred between
        # the two known observations.
        # ----------------------------------------------------

        between = race_driver[
            race_driver["LapNumber"].between(
                previous["LapNumber"],
                following["LapNumber"],
                inclusive="neither",
            )
        ]

        has_pit_activity = (
            between["PitInTime"].notna().any()
            or between["PitOutTime"].notna().any()
        )

        candidates.append(
            {
                "Year": block["Year"],
                "GrandPrix": block["GrandPrix"],
                "Driver": block["Driver"],

                "missing_stint":
                    block["Stint"],

                "start_lap":
                    block["start_lap"],

                "end_lap":
                    block["end_lap"],

                "n_missing":
                    block["n_missing"],

                "previous_lap":
                    previous["LapNumber"],

                "previous_stint":
                    previous["Stint"],

                "previous_compound":
                    previous_compound,

                "previous_tyre_life":
                    previous["TyreLife"],

                "next_lap":
                    following["LapNumber"],

                "next_stint":
                    following["Stint"],

                "next_compound":
                    next_compound,

                "next_tyre_life":
                    following["TyreLife"],

                "lap_gap":
                    lap_gap,

                "tyre_gap":
                    tyre_gap,

                "compound_changed":
                    compound_changed,

                "has_pit_activity":
                    has_pit_activity,

                "sequence_consistent":
                    tyre_gap == lap_gap,
            }
        )

    return pd.DataFrame(candidates)

def find_compound_change_missing_tyre_life(df):
    """
    Find cases where a driver changes compound and the new compound
    has missing TyreLife values followed by a known TyreLife value.

    These are potential candidates for reconstructing TyreLife
    from the beginning of the new tyre sequence.
    """

    df = df.copy()

    df["TyreLife"] = pd.to_numeric(
        df["TyreLife"],
        errors="coerce",
    )

    df = df.sort_values(
        [
            "Year",
            "GrandPrix",
            "Driver",
            "LapNumber",
        ]
    )

    candidates = []

    group_columns = [
        "Year",
        "GrandPrix",
        "Driver",
    ]

    for _, race_driver in df.groupby(
        group_columns,
        dropna=False,
    ):

        race_driver = race_driver.sort_values(
            "LapNumber"
        ).copy()

        rows = race_driver.reset_index(
            drop=True
        )

        for i in range(
            1,
            len(rows),
        ):

            current = rows.iloc[i]
            previous = rows.iloc[i - 1]

            # ------------------------------------------------
            # We are interested in an actual compound change.
            # ------------------------------------------------

            if (
                pd.isna(previous["Compound"])
                or pd.isna(current["Compound"])
            ):
                continue

            if (
                previous["Compound"]
                == current["Compound"]
            ):
                continue

            # ------------------------------------------------
            # Current lap must have missing TyreLife.
            # ------------------------------------------------

            if pd.notna(
                current["TyreLife"]
            ):
                continue

            new_compound = current["Compound"]

            # ------------------------------------------------
            # Find consecutive laps with the new compound
            # and missing TyreLife.
            # ------------------------------------------------

            missing_rows = []

            j = i

            while j < len(rows):

                row = rows.iloc[j]

                if row["Compound"] != new_compound:
                    break

                if pd.notna(row["TyreLife"]):
                    break

                missing_rows.append(row)

                j += 1

            if not missing_rows:
                continue

            # ------------------------------------------------
            # We need a known TyreLife after the missing block.
            # ------------------------------------------------

            if j >= len(rows):
                continue

            next_row = rows.iloc[j]

            if (
                next_row["Compound"]
                != new_compound
            ):
                continue

            if pd.isna(
                next_row["TyreLife"]
            ):
                continue

            candidates.append(
                {
                    "Year":
                        current["Year"],

                    "GrandPrix":
                        current["GrandPrix"],

                    "Driver":
                        current["Driver"],

                    "previous_compound":
                        previous["Compound"],

                    "new_compound":
                        new_compound,

                    "compound_change_lap":
                        previous["LapNumber"],

                    "first_missing_lap":
                        missing_rows[0]["LapNumber"],

                    "last_missing_lap":
                        missing_rows[-1]["LapNumber"],

                    "n_missing":
                        len(missing_rows),

                    "next_known_lap":
                        next_row["LapNumber"],

                    "next_tyre_life":
                        next_row["TyreLife"],

                    "has_pit_in":
                        any(
                            pd.notna(
                                row["PitInTime"]
                            )
                            for row in missing_rows
                        ),

                    "has_pit_out":
                        any(
                            pd.notna(
                                row["PitOutTime"]
                            )
                            for row in missing_rows
                        ),
                }
            )

    return pd.DataFrame(candidates)

def repair_compound_change_tyre_life(df):
    """
    Repair TyreLife for a new compound when the missing sequence
    is bounded by a known TyreLife value and the sequence is
    internally consistent.

    Example:

        MEDIUM  8
        MEDIUM  9
        PIT
        HARD   NaN
        HARD   NaN
        HARD    3

    becomes:

        MEDIUM  8
        MEDIUM  9
        PIT
        HARD    1
        HARD    2
        HARD    3

    Original values are preserved for auditing.
    """

    df_fixed = df.copy()

    # --------------------------------------------------------
    # Normalize TyreLife
    # --------------------------------------------------------

    df_fixed["TyreLife"] = pd.to_numeric(
        df_fixed["TyreLife"],
        errors="coerce",
    )

    # Preserve original values.
    df_fixed["TyreLifeOriginal"] = (
        df_fixed["TyreLife"]
    )

    # Repair metadata.
    df_fixed["TyreLifeWasRepaired"] = False

    df_fixed["TyreLifeRepairMethod"] = (
        pd.Series(
            pd.NA,
            index=df_fixed.index,
            dtype="string",
        )
    )

    # --------------------------------------------------------
    # Process each driver within each race.
    # --------------------------------------------------------

    group_columns = [
        "Year",
        "GrandPrix",
        "Driver",
    ]

    for _, race_driver in df_fixed.groupby(
        group_columns,
        dropna=False,
    ):

        race_driver = race_driver.sort_values(
            "LapNumber"
        )

        indices = race_driver.index.tolist()

        for position in range(
            1,
            len(indices),
        ):

            current_idx = indices[position]
            previous_idx = indices[position - 1]

            current = df_fixed.loc[
                current_idx
            ]

            previous = df_fixed.loc[
                previous_idx
            ]

            # ------------------------------------------------
            # We need a compound change.
            # ------------------------------------------------

            if (
                pd.isna(previous["Compound"])
                or pd.isna(current["Compound"])
            ):
                continue

            if (
                previous["Compound"]
                == current["Compound"]
            ):
                continue

            # ------------------------------------------------
            # The first lap of the new compound must have
            # missing TyreLife.
            # ------------------------------------------------

            if pd.notna(
                current["TyreLife"]
            ):
                continue

            new_compound = current["Compound"]

            # ------------------------------------------------
            # Collect consecutive missing laps belonging
            # to the new compound.
            # ------------------------------------------------

            missing_indices = []

            j = position

            while j < len(indices):

                idx = indices[j]

                row = df_fixed.loc[idx]

                if (
                    row["Compound"]
                    != new_compound
                ):
                    break

                if pd.notna(
                    row["TyreLife"]
                ):
                    break

                missing_indices.append(idx)

                j += 1

            if not missing_indices:
                continue

            # ------------------------------------------------
            # Need a known TyreLife after the missing block.
            # ------------------------------------------------

            if j >= len(indices):
                continue

            next_idx = indices[j]

            next_row = df_fixed.loc[
                next_idx
            ]

            if (
                next_row["Compound"]
                != new_compound
            ):
                continue

            if pd.isna(
                next_row["TyreLife"]
            ):
                continue

            next_tyre_life = (
                next_row["TyreLife"]
            )

            n_missing = len(
                missing_indices
            )

            # ------------------------------------------------
            # The expected final TyreLife is exactly equal
            # to the number of missing laps + 1.
            #
            # Example:
            #
            # 2 missing laps
            # followed by TyreLife = 3
            #
            # => missing values must be 1 and 2.
            # ------------------------------------------------

            expected_next_tyre_life = (
                n_missing + 1
            )

            if (
                next_tyre_life
                != expected_next_tyre_life
            ):
                continue

            # ------------------------------------------------
            # Repair the missing sequence.
            # ------------------------------------------------

            for offset, idx in enumerate(
                missing_indices,
                start=1,
            ):

                df_fixed.loc[
                    idx,
                    "TyreLife"
                ] = offset

                df_fixed.loc[
                    idx,
                    "TyreLifeWasRepaired"
                ] = True

                df_fixed.loc[
                    idx,
                    "TyreLifeRepairMethod"
                ] = (
                    "compound_change_sequence"
                )

    return df_fixed

def validate_tyre_life_repairs(repaired_df,):
    """
    Validate TyreLife repairs.
    """

    repaired_mask = (
        repaired_df["TyreLifeWasRepaired"]
    )

    n_repaired = repaired_mask.sum()

    print(
        f"Repaired TyreLife values: "
        f"{n_repaired:,}"
    )

    if n_repaired == 0:
        print(
            "No TyreLife values were repaired."
        )
        return

    print(
        "\nRepair methods:"
    )

    print(
        repaired_df.loc[
            repaired_mask,
            "TyreLifeRepairMethod",
        ]
        .value_counts()
    )

    print(
        "\nExamples of repaired rows:"
    )

    print(
        repaired_df.loc[
            repaired_mask,
            [
                "Year",
                "GrandPrix",
                "Driver",
                "LapNumber",
                "Compound",
                "TyreLifeOriginal",
                "TyreLife",
                "TyreLifeRepairMethod",
            ],
        ]
        .head(30)
        .to_string(index=False)
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    df = load_processed_laps()

    repaired_df = repair_compound_change_tyre_life(df)

    repaired_df["HasTyreLife"] = (
        repaired_df["TyreLife"].notna()
    )

    print(
        "\nRepair methods:"
    )

    print(
        repaired_df["TyreLifeRepairMethod"]
        .value_counts(dropna=False)
    )

    print(
        "\nExamples of repaired rows:"
    )

    print(
        repaired_df[
            repaired_df["TyreLifeWasRepaired"]
        ][
            [
                "Year",
                "GrandPrix",
                "Driver",
                "LapNumber",
                "Compound",
                "TyreLifeOriginal",
                "TyreLife",
                "TyreLifeRepairMethod",
            ]
        ]
        .head(30)
        .to_string(index=False)
    )

