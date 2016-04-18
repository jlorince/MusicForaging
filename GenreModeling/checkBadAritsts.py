import pylast
import pandas as pd
from urllib import quote_plus,unquote_plus
import sys

API_KEY,API_SECRET = open('../lastfm.apikey').readlines()
outfile = 'data/lastfm_bad_artists'

network = pylast.LastFMNetwork(api_key=API_KEY, api_secret=API_SECRET)

artists = pd.read_table("data/vocab_idx",header=None,names=['artist','idx'])

start = 84079
with open(outfile,'a') as fout:
    for i,a in enumerate(artists['artist']):
        if i<start:
            continue
        print i,a
        artist = network.get_artist(unquote_plus(a))
        try:
            bio = artist.get_bio_content()
        except:
            continue
        if "This is not an artist, but appears here due to incorrectly tagged tracks." in bio:
            fout.write(a+'\n')
            fout.flush()
