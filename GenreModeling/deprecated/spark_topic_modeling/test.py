# sudo apt-get install emacs
# sudo apt-get install less
# sudo apt-get install screen
# gsutil cp gs://music-foraging/vocab_idx .

# usage python test.py artist_topic_file top_terms

import numpy as np
import sys
fi = sys.argv[1]
n_terms = int(sys.argv[2])

data = np.loadtxt(fi)
data = data/data.sum(0)


d = {int(idx):term for term,idx in [line.strip().split('\t') for line in open('vocab_idx').readlines()]}

for topic in xrange(data.shape[1]):
    print [d[i] for i in np.argsort(data[:,topic])[::-1][:n_terms]]
