"""Download NASA POWER daily gridded meteorology, retaining raw responses."""
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone
import requests
import pandas as pd
import numpy as np
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).parent / 'data' / 'new_delhi'
ROOT.mkdir(parents=True, exist_ok=True)
session = requests.Session()
session.mount('https://', HTTPAdapter(max_retries=Retry(total=4, backoff_factor=2, status_forcelist=[429,500,502,503,504])))
records = []; sources = []
for start, end in [(1985,1994),(1995,2004),(2005,2014),(2015,2025)]:
    params = dict(parameters='T2M_MAX,T2M_MIN,RH2M', community='AG', longitude=77.2090, latitude=28.6139,
                  start=f'{start}0101', end=f'{end}1231', format='JSON', **{'time-standard':'LST'})
    path = ROOT / f'power_{start}_{end}.json'
    request = requests.Request('GET','https://power.larc.nasa.gov/api/temporal/daily/point',params=params).prepare()
    if not path.exists():
        print(f'Downloading {start}–{end}', flush=True)
        response = session.get(request.url, timeout=240); response.raise_for_status()
        raw = response.json()
        if 'properties' not in raw: raise ValueError(raw)
        path.write_text(response.text)
    raw = json.loads(path.read_text())
    df = pd.DataFrame(raw['properties']['parameter'])
    df.index = pd.to_datetime(df.index, format='%Y%m%d')
    df = df.replace(raw.get('header',{}).get('fill_value', -999), np.nan)
    records.append(df)
    sources.append({'url':request.url, 'raw_file':path.name, 'sha256':hashlib.sha256(path.read_bytes()).hexdigest(), 'header':raw.get('header'), 'parameters':raw.get('parameters')})
weather = pd.concat(records).sort_index().rename(columns={'T2M_MAX':'max_temp_c','T2M_MIN':'min_temp_c','RH2M':'relative_humidity_pct'})
expected = pd.date_range('1985-01-01','2025-12-31')
if not weather.index.equals(expected) or weather.isna().any().any():
    raise ValueError('Incomplete dates or missing measurements; inspect raw data before proceeding.')
weather.index.name = 'date'
weather.to_csv(ROOT / 'weather_1985_2025.csv')
baseline = weather.loc['1985':'2014']
# Smooth across circular calendar days; map all dates to leap year 2000.
def calendar_day(index):
    return pd.to_datetime('2000-' + index.strftime('%m-%d')).dayofyear
base_days = calendar_day(baseline.index)
normal = {}; q90 = {}; q975 = {}
for day in range(1,367):
    distance = np.abs(base_days-day); values = baseline.loc[np.minimum(distance,366-distance)<=7,'max_temp_c']
    normal[day] = float(values.mean()); q90[day] = float(values.quantile(.90)); q975[day] = float(values.quantile(.975))
climatology = pd.DataFrame({'calendar_day':range(1,367),'normal_max_temp_c':list(normal.values()),'p90_max_temp_c':list(q90.values()),'p97_5_max_temp_c':list(q975.values())})
climatology.to_csv(ROOT / 'climatology_1985_2014.csv',index=False)
df = weather.loc['2015':'2025'].copy(); days = calendar_day(df.index)
df['normal_max_temp_c'] = [normal[d] for d in days]
df['temp_anomaly_c'] = df.max_temp_c-df.normal_max_temp_c
p90 = np.array([q90[d] for d in days]); p975 = np.array([q975[d] for d in days])
# Research proxy labels: four explicit temperature categories, not official alerts.
heat = (df.max_temp_c>=40) & ((df.temp_anomaly_c>=4.5) | (df.max_temp_c>=45))
severe = (df.max_temp_c>=40) & ((df.temp_anomaly_c>=6.5) | (df.max_temp_c>=47))
# Absolute thresholds apply even if anomaly is small.
heat |= df.max_temp_c>=45; severe |= df.max_temp_c>=47
# Moderate is an invented research category; Extreme also includes local p97.5 exceedance
# with Tmax >=40, to support four-class research without pretending it is IMD severe.
extreme = severe | ((df.max_temp_c>=40) & (df.max_temp_c>=p975))
moderate = (df.max_temp_c>=p90) | (df.max_temp_c>=38)
df['risk'] = np.select([extreme,heat,moderate],['Extreme','High','Moderate'],default='Low')
# Counter of consecutive days above local 90th percentile (past/current days only).
run=0; durations=[]
for hot in (df.max_temp_c>=p90):
    run=run+1 if hot else 0
    durations.append(max(1,run))
df['duration_days']=durations
df['above_local_p90']=(df.max_temp_c>=p90).to_numpy()
df['imd_like_heat_day']=heat.to_numpy(); df['imd_like_severe_day']=severe.to_numpy()
df.to_csv(ROOT / 'delhi_2015_2025_labeled.csv')
metadata = {'location':'New Delhi, India','latitude':28.6139,'longitude':77.2090,'retrieved_utc':datetime.now(timezone.utc).isoformat(),
 'source':'NASA POWER gridded meteorological estimates, not station observations','reference_period':'1985–2014','study_period':'2015–2025','rows':len(df),'class_counts':df.risk.value_counts().to_dict(),
 'climatology_method':'Calendar-day +/-7-day circular window on 30 historical years, using leap-year calendar',
 'labels':'Research-derived temperature hazard proxies, not official IMD declarations or observed health outcomes. Moderate: Tmax >= local p90 OR >=38 C. High: Tmax >=40 C and anomaly >=4.5 C, OR Tmax>=45 C. Extreme: Tmax>=40 C and anomaly>=6.5 C, OR Tmax>=47 C, OR Tmax>=40 C and >=local p97.5. Higher class takes precedence.',
 'duration':'Consecutive current/past days above local p90; non-exceedance mapped to 1 for model schema. January 2015 starts a fresh counter.',
 'sources':sources}
(ROOT / 'provenance.json').write_text(json.dumps(metadata,indent=2))
print(json.dumps({'rows':len(df),'counts':metadata['class_counts']},indent=2))
