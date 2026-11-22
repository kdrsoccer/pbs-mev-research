"""Load parquet, dedupe on slot, derive correct UTC time from slot, save slim copy."""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); import paths
import pandas as pd, numpy as np
SRC=paths.SRC
OUT=paths.WORK+'slim.parquet'
cols=['slot','builder_pubkey','relay','value_wei','value_eth','n_bids','n_builders','n_relays',
      'top_bid_wei','second_bid_wei','median_bid_wei','earliest_bid_ms','latest_bid_ms','bid_margin',
      'datetime','hour','dayofweek','timestamp','proposer_fee_recipient']
full=pd.read_parquet(SRC)
full.sort_values(['slot','value_eth'],ascending=[True,False]).drop_duplicates('slot',keep='first').reset_index(drop=True).to_parquet(paths.WORK+'dedup.parquet',index=False)  # all columns, deduped: input to 3_figures/
df=full[cols].copy(); del full
print('raw rows',len(df))
print('unique slots',df.slot.nunique())
dups=df[df.slot.duplicated(keep=False)].sort_values('slot')
print('dup rows',len(dups)); print(dups.head(10)[['slot','relay','value_eth','n_bids']])
# are dup rows identical in value?
g=dups.groupby('slot').value_eth.nunique(); print('dup slots with differing value:',(g>1).sum(), 'of', len(g))
# relay counts raw and deduped
print('relay counts raw\n',df.relay.value_counts())
df=df.sort_values(['slot','value_eth'],ascending=[True,False]).drop_duplicates('slot',keep='first').reset_index(drop=True)
print('after dedupe',len(df))
print('relay counts dedup\n',df.relay.value_counts(), df.relay.value_counts(normalize=True))
# time from slot
df['t']=pd.to_datetime(1606824023+12*df.slot,unit='s',utc=True)
print('time range',df.t.min(),df.t.max())
print('stored datetime range',df.datetime.min(),df.datetime.max())
print('stored timestamp range',df.timestamp.min(),df.timestamp.max())
off=(df.t - pd.to_datetime(df.datetime,utc=True)).dt.total_seconds()/86400
print('offset days stored vs slot-derived: ',off.describe())
df['hour_c']=df.t.dt.hour; df['dow_c']=df.t.dt.dayofweek
print('stored hour == correct hour frac',(df.hour==df.hour_c).mean(),' dow eq',(df.dayofweek==df.dow_c).mean())
print('matched (n_bids notna)',df.n_bids.notna().sum(), df.n_bids.notna().mean())
print('value stats',df.value_eth.describe(), 'max',df.value_eth.max())
print('n unique builders',df.builder_pubkey.nunique())
print('proposer_fee_recipient nunique',df.proposer_fee_recipient.nunique())
df.drop(columns=['datetime','hour','dayofweek','timestamp','proposer_fee_recipient']).to_parquet(OUT,index=False)
