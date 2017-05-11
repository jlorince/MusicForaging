import pylast
import pandas as pd
from urllib.parse import quote_plus,unquote_plus


API_KEY,API_SECRET = open('../lastfm.apikey').readlines()
outfile = 'data/lastfm_top_similar_artists_new'

network = pylast.LastFMNetwork(api_key=API_KEY, api_secret=API_SECRET)

artists = pd.read_table("data/vocab_idx",header=None,names=['artist','idx'])
all_artist_names = set(artists['artist'])

done = set()
try:
    with open(outfile) as fin:
        for line in fin:
            done.add(line[:line.find('\t')])
except IOError:
    pass


with open(outfile,'a') as out:
    for i,a in enumerate(artists['artist']):
        print i,a
        while True:
            if a in done:
                break
            try:
                artist = network.get_artist(unquote_plus(a))
                result = artist.get_similar(limit=1000)
                names = [quote_plus(r.item.name.encode('utf8')).lower() for r in result]
                print names
                out.write(a+'\t'+' '.join(names)+'\n')
                #coverage = len(all_artist_names.intersection(names))
                break
            except pylast.WSError as e:
                print e
                if "The artist you supplied could not be found"==str(e):
                    out.write(a+'\n')
                    break
            except pylast.MalformedResponseError as e:
                if "Errno 10054" in e.message:
                    print e
                    print "trying again..."
                    time.sleep(10)
            except pylast.NetworkError:
                if "connection attempt" in str(e):
                    print e
                    print "trying again..."
                    time.sleep(10)


