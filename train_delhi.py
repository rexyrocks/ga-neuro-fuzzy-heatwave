import pandas as pd
from heatwave import train
frame = pd.read_csv('data/new_delhi/delhi_2015_2025_labeled.csv')
frame.attrs['origin'] = 'NASA POWER New Delhi gridded daily meteorology 2015–2025; research-derived temperature hazard labels; climatology 1985–2014. See data/new_delhi/provenance.json.'
train(frame, 'results_delhi', chronological=True)
