import sys
import glob
import os

logfi = sys.argv[1]

if os.path.isdir(logfi):
    files = glob.glob(logfi+'*')
else:
    files = [logfi]

done = set()
for fi in files:
    with open(fi) as fin:
        for line in fin:
            if 'User' in line:
                filename = line.strip().split()[-1]
                done.add(filename[filename.rfind('/'):-1])

with open('joblist.txt') as fin, open('joblist_partial.txt','w') as fout:
    for line in fin:
        line = line.strip()
        if line[line.rfind('/'):] in done:
            continue
        else:
            fout.write(line+'\n')
