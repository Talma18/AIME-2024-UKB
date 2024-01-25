import pandas as pd
from typing import List
from dataclasses import dataclass

MIN_DURATION_WEEKS = 6
MAX_PAUSE_WEEKS = 14
ICD_FN = "../disease_onset.csv"
ATC_FN = (
    "../drug_export/out/ukb_presner.pkl"
)


@dataclass
class Therapy:
    start: pd.Timestamp
    end: pd.Timestamp
    drug: str

    def __post_init__(self):
        self.sort_index = self.start

    @property
    def length_d(self):
        """Length in days"""
        try:
            return (self.end - self.start).days
        except ValueError:
            return 0

    @property
    def length_w(self):
        """Length in weeks"""
        return float((self.end - self.start).days / 7)

    def overlaps(self, other: "Therapy"):
        """Returns true if the two therapies overlap"""
        return self.start <= other.end and self.end >= other.start

    def is_continuation_of(self, other: "Therapy"):
        """Returns true if the two therapies are consecutive"""
        dif_days = (other.start - self.end).days
        dif_weeks = dif_days / 7
        return dif_weeks <= MAX_PAUSE_WEEKS

    def is_adequate_switch(self, other: "Therapy"):
        """
        Returns true if the two therapies:
        - use different drug
        - are consecutive
        - are non overlapping
        """
        if self.drug == other.drug:
            return False  # same drug
        if not self.is_continuation_of(other):
            return False  # not consecutive (longer break in therapy)
        if self.overlaps(other):
            return False  # overlapping (combination therapy)
        return True

    def __repr__(self):
        start_date = self.start.strftime("%Y-%m-%d")
        end_date = self.end.strftime("%Y-%m-%d")
        return f"Therapy({self.drug}: {start_date} - {end_date})"

    # use start for sorting
    def __lt__(self, other):
        return self.sort_index < other.sort_index


def main():
    # atc: pd.DataFrame = pd.read_csv(ATC_FN, index_col=0)
    atc: pd.DataFrame = pd.read_pickle(ATC_FN)
    atc["issue_date"] = pd.to_datetime(atc["issue_date"])
    print("With prescription data: ", atc.index.nunique())

    # adeq diagnosis profile is required for TRDstatus
    atc = filter_atc_to_ppl_with_adeq_depr_diag(atc)
    print("With adeq depr diag and prescription data: ", atc.index.nunique())

    df = get_ADprescriptionDateL_df(atc)
    df = df.reset_index().set_index("eid")  # move `chemblid` from multi-index to col
    df["therapy_L"] = df.apply(
        lambda row: convert_dateL_to_therapyL(row["chemblid"], row["issue_date"]),
        axis=1,
    )

    # collapse to one row per person
    df = df.reset_index()
    df = df.groupby("eid").agg({"therapy_L": list})
    # therapy lists for different ATC-codes collapsed into nested list
    # flatten therapy_L
    df["therapy_L"] = df["therapy_L"].apply(lambda L: sorted([t for l in L for t in l]))
    # by going through  list, count adequate switches
    df["adeq_switch_count"] = df.apply(
        lambda row: count_adeq_therapy_switches(row["therapy_L"]), axis=1
    )

    df["trd"] = df["adeq_switch_count"] >= 2

    # if no serious therapy was tried, we can't say anything about TRDstatus
    # count switches, including non-adequate ones
    df["therapy_count"] = df["therapy_L"].apply(len)
    l_bef = df.shape[0]
    # drop ppl with no adeq-therapy
    df = df.loc[df["therapy_count"] > 0]
    l_after = df.shape[0]
    print("Lost due no adeq therapy", l_bef - l_after)

    print(df["adeq_switch_count"].value_counts())

    print(df["trd"].value_counts())

    df = df[['trd', 'adeq_switch_count', 'therapy_count', 'therapy_L']]
    df.to_csv(f"trd.chembl.csv")
    return df


def convert_dateL_to_therapyL(
    atc: str,
    date_L: List[pd.Timestamp],
    min_duration_weeks: int = MIN_DURATION_WEEKS,
    max_pause_weeks: int = MAX_PAUSE_WEEKS,
) -> List[Therapy]:
    """
    Given a list of dates for an ATC-code, return instances of the `Therapy` class.

    - `min_duration_weeks`: minimum duration of the therapy in weeks,
    below this, the therapy is discarded
    - `max_pause_weeks`: maximum pause between two prescriptions in weeks,
    above this, the therapy is split into two
    """
    date_L = sorted(date_L)
    if len(date_L) == 0:
        return []
    max_pause_days = max_pause_weeks * 7
    min_duration_days = min_duration_weeks * 7

    therapy_L = []
    start = date_L[0]
    last = date_L[0]
    for date in date_L[1:]:
        dif_days = (date - last).days
        if dif_days > max_pause_days:
            # Pause too long, end therapy
            therapy_L.append((start, last))
            start = date
        last = date
    therapy_L.append((start, last))

    therapy_L = [Therapy(start=t[0], end=t[1], drug=atc) for t in therapy_L]
    therapy_L = [t for t in therapy_L if t.length_d >= min_duration_days]
    return therapy_L


def filter_atc_to_ppl_with_adeq_depr_diag(atc: pd.DataFrame) -> pd.Index:
    """
    Filter the `atc` dataframe to people with adequate depression diagnosis.

    Prescription records of people who are not diagnosed with depression/have conflicting diagnosis are discarded.
    """
    global ICD_FN

    icd = pd.read_csv(ICD_FN, index_col=0)
    idx = icd.index.intersection(atc.index)
    icd = icd.loc[idx]
    atc = atc.loc[idx]

    # RELEVANT DISEASE COLS
    tgt_cols = ["F31", "F32", "F33", "F34"]
    tgt_cols = [c for c in tgt_cols if c in icd.columns]
    # F10-F19  Mental and behavioral disorders due to psychoactive substance use
    f10f19 = [f"F{i}" for i in range(10, 20) if f"F{i}" in icd.columns]
    # F20-F29  Schizophrenia, schizotypal, delusional, and other non-mood psychotic disorders
    f20f29 = [f"F{i}" for i in range(20, 30) if f"F{i}" in icd.columns]
    use_cols = tgt_cols + f10f19 + f20f29

    # drop unused cols
    icd = icd[use_cols]
    l0 = icd.shape[0]

    # keep ppl with some kind of depression
    icd = icd.loc[(icd[["F32", "F33", "F34"]] > 0).any(axis=1)]
    l1 = icd.shape[0]
    print("Lost due no depressive diag", l0 - l1)

    # drop ppl with conflicting diagnosis
    icd = icd.loc[~(icd[[*f10f19, *f20f29, "F31"]] > 0).any(axis=1)]
    l2 = icd.shape[0]
    print("Lost due conflicting diag", l1 - l2)
    print("Meets Fabbri et al depression criteria", icd.shape[0])

    return atc.loc[atc.index.intersection(icd.index)]


def get_ADprescriptionDateL_df(atc: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a dataframe with prescription data. Filters for antidepressants.
    Aggregates prescription dates into a separate list for every person (ID), per ATC-code.
    """
    global MIN_DURATION_WEEKS, MAX_PAUSE_WEEKS, ICD_FN, ATC_FN

    def _filter_antidepressants(df: pd.DataFrame):
        # return df.loc[df["atc_code"].str.startswith("N06A")]
        d = pd.read_pickle("AD.chembl_atc_D.pkl")
        return df.loc[df["chemblid"].isin(d)]


    def _per_id_per_atc_dateL(df: pd.DataFrame) -> pd.DataFrame:
        """
        For every person (ID) and every different ATC-code,
        get the dates of the prescriptions into a separate list.
        """
        return df.groupby(["eid", "chemblid"]).agg({"issue_date": list})

    # antidepressant consumption is required for TRDstatus
    len_bef = atc.index.nunique()
    atc = _filter_antidepressants(atc)
    len_aft = atc.index.nunique()
    print("Lost due to not taking any AD", len_bef - len_aft)
    df = _per_id_per_atc_dateL(atc)
    return df


def count_adeq_therapy_switches(therapy_L: List[Therapy]):
    """
    Given a list of therapies, count how many times the therapy switches
    adequately (continous, non overlapping treatment with different drug).
    """
    therapy_L = sorted(therapy_L, key=lambda t: t.start)
    count = 0
    for i, t in enumerate(therapy_L):
        if i == 0:
            continue
        if t.is_adequate_switch(therapy_L[i - 1]):
            count += 1
    return count


def _test_therapies():
    """
    This function is purely for testing purposes.
    """
    ts = pd.Timestamp("2012-01-01 00:00:00")
    week = pd.Timedelta("7 days")

    l = [ts]
    print("Too short", convert_dateL_to_therapyL(l))

    l = [ts + i * week for i in range(0, 30)]
    print("weekly", convert_dateL_to_therapyL(l))

    l = [ts, ts + week * 7]
    print("7week", convert_dateL_to_therapyL(l))

    l = [ts, ts + week * 20]
    print("too long pause", convert_dateL_to_therapyL(l))

    l = [ts, ts + week * 7, ts + week * 50, ts + week * 57]
    print(l)
    print("two therapies", convert_dateL_to_therapyL(l))

    l1 = [ts + i * week for i in range(0, 30)]
    l2 = [ts + i * week for i in range(50, 70)]
    l = l1 + l2
    print("two weekly", convert_dateL_to_therapyL(l))
    return


if __name__ == "__main__":
    main()
