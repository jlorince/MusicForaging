import sys

logfi = sys.argv[1]

done = set()
with open(logfi) as fin:
    for line in fin:
        if 'User' in line:
            filename = line.strip().split()[-1]
            done.add(filename[filename.rfind('/'):-1])

with open('joblist.txt') as fin, open('joblist_partial.txt','w') as fout:
    for line in fin:
        line = line.strip()
        break
        if line[line.rfind('/'):] in done:
            continue
        else:
            fout.write(line+'\n')

