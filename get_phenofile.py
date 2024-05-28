infile1 = open('training_STDB.samples', 'rt')
infile2 = open('phenotyped.tsv', 'rt')
outfile = open('training_STDB.phenos', 'wt')

line = infile2.readline()

d = {}

for line in infile2:
	line = line.rstrip()
	split = line.split('\t')
	if split[0] == 'NA' and split[4] != 'unknown':
		if split[4] == 'mare':
			d[split[1]] = 1
		elif split[4] == 'gelding':
			d[split[1]] = 0
		elif split[4] == 'stallion':
			d[split[1]] = 0
		elif split[4] == 'male':
			d[split[1]] = 0
		elif split[4] == 'colt':
			d[split[1]] = 0
		elif split[4] == 'Ridgling':
			d[split[1]] = 0
		else:
			print(split[4])
	else:
		if split[0] != 'NA' and split[4] != 'unknown':
			if split[4] == 'mare':
				d[split[0]] = 1
			elif split[4] == 'gelding':
				d[split[0]] = 0
			elif split[4] == 'stallion':
				d[split[0]] = 0
			elif split[4] == 'male':
				d[split[0]] = 0
			elif split[4] == 'colt':
				d[split[0]] = 0
			elif split[4] == 'Ridgling':
				d[split[0]] = 0
			else:
				print(split[4])

print(len(d))

d2 = {}

infile2 = open('/panfs/jay/groups/27/mccuem/dimml002/NuGEN/Sample_Organization/Spring2024/phenotyped.tsv', 'rt')

line = infile2.readline()

for line in infile2:
	line = line.rstrip()
	split = line.split('\t')
	if split[0] == 'NA':
		if split[5] == 'control':
			d2[split[1]] = 0
		elif split[5] == 'case':
			d2[split[1]] = 1
		else:
			print(split[0])
	else:
		if split[5] == 'control':
			d2[split[0]] = 0
		elif split[5] == 'case':
			d2[split[0]] = 1
		else:
			print(split[0])

print(len(d2))
print(d['M6116'])


d3 = {}

infile3 = open('phenotyped_STDBs.tsv', 'rt')

line = infile3.readline()

for line in infile3:
	line = line.rstrip()
	split = line.split('\t')
	if split[0] == 'NA':
		if split[7] == 'trotter':
			d3[split[1]] = 1
		elif split[7] == 'pacer':
			d3[split[1]] = 0
		elif split[7] == 'pacer/trotter':
			d3[split[1]] = 0
		else:
			print(split[7])
	else:
		if split[7] == 'trotter':
			d3[split[0]] = 1
		elif split[7] == 'pacer':
			d3[split[0]] = 0
		elif split[7] == 'pacer/trotter':
			d3[split[0]] = 0
		else:
			print(split[7])

for line in infile1:
	line = line.rstrip()
	split = line.split('\t')
	if split[0] in d:
		#print(str(d2[split[0]]) + ' ' + split[0] + ' ' + str(d[split[0]]) + ' ' + str(d3[split[0]]), file=outfile)
		print(str(d2[split[0]]) + ' ' + split[0] + ' ' + str(d[split[0]]) + ' ' + str(d3[split[0]]), file=outfile)
