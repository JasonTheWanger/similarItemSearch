# split_groups.py
import pandas as pd, numpy as np

df = pd.read_csv("shopee-product-matching/train.csv")  # must have 'image' and 'label_group'

# group sizes
sizes = df.groupby("label_group").size()
multi = sizes[sizes >= 2].index.tolist()
single = sizes[sizes == 1].index.tolist()

rng = np.random.default_rng(42)

def slice_groups(groups, ratios=(0.70, 0.15, 0.15)):
    n = len(groups)
    n_train = int(round(ratios[0]*n))
    n_val   = int(round(ratios[1]*n))
    return set(groups[:n_train]), set(groups[n_train:n_train+n_val]), set(groups[n_train+n_val:])

def take(df, groups): return df[df["label_group"].isin(groups)].copy()

def stats(name, d):
    gs = d.groupby("label_group").size()
    return dict(rows=len(d), groups=len(gs), multi=(gs>=2).sum(), single=(gs==1).sum())

# --- resample until val/test have enough multi groups ---
target_val_multi  = max(50, int(0.12 * len(multi)))   # tweak if your dataset is small
target_test_multi = max(50, int(0.12 * len(multi)))

attempts = 0
while True:
    attempts += 1
    m = multi.copy(); s = single.copy()
    rng.shuffle(m); rng.shuffle(s)

    m_tr, m_va, m_te = slice_groups(m, (0.70, 0.15, 0.15))
    s_tr, s_va, s_te = slice_groups(s, (0.70, 0.15, 0.15))

    df_train = pd.concat([take(df, m_tr), take(df, s_tr)], ignore_index=True)
    df_val   = pd.concat([take(df, m_va), take(df, s_va)], ignore_index=True)
    df_test  = pd.concat([take(df, m_te), take(df, s_te)], ignore_index=True)

    # drop singletons from TRAIN (no positives there)
    df_train = df_train.groupby("label_group").filter(lambda g: len(g) >= 2).reset_index(drop=True)

    st_tr, st_va, st_te = stats("TRAIN", df_train), stats("VAL", df_val), stats("TEST", df_test)

    if st_va["multi"] >= target_val_multi and st_te["multi"] >= target_test_multi:
        print(f"Split OK after {attempts} attempt(s).")
        print("TRAIN:", st_tr, "\nVAL:  ", st_va, "\nTEST: ", st_te)
        df_train.to_csv("shopee-product-matching/train_groups.csv", index=False)
        df_val.to_csv("shopee-product-matching/val_groups.csv", index=False)
        df_test.to_csv("shopee-product-matching/test_groups.csv", index=False)
        break
