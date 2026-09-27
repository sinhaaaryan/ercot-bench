import time, pandas as pd
from gridstatus import Ercot
e=Ercot()
x=pd.ExcelFile("data/raw/fuel_mix/IntGenbyFuel2024.xlsx"); df=x.parse("Mar")
d=df[df.Date.astype(str).str.startswith("2024-03-10")].iloc[0]; print("spring nan cols:", [c for c in df.columns if pd.isna(d[c])])
for fn in ["get_dam_spp","get_rtm_spp"]:
    t=time.time(); s=getattr(e,fn)(2024, verbose=False)
    print(fn, f"{time.time()-t:.0f}s", s.shape, s.dtypes.to_dict())
    print(s.head(3).to_string())
    lo=s[s["Location"].str.match(r"^(HB|LZ)_")]; print(sorted(lo.Location.unique()))
    print(lo[(lo.Location=="HB_HOUSTON")&(lo["Interval Start"].astype(str).str.startswith("2024-11-03 0"))].to_string()[:2500])
    s.to_pickle(f"data/raw/probe_{fn}.pkl")
