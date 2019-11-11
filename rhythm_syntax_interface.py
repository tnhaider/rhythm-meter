import sys, re, os
import joblib, json
import string
from numpy import mean

import numpy as np
import matplotlib.pyplot as plt

from nltk.tokenize import RegexpTokenizer
import pyphen

from collections import Counter


dic = pyphen.Pyphen(lang='de_DE')
tokenizer = RegexpTokenizer(r'\w+')

def word2features(sentence, index):
        word = sentence[index][0]
        postag = sentence[index][1]
        features = {
        # uebernommen vom DecisionTreeClassifier
                'word': word,
                'position_in_sentence': index,
                'rel_position_in_sentence': index / len(sentence),
                'is_first': index == 0,
                'is_last': index == len(sentence) - 1,
                'is_capitalized': word[0].upper() == word[0],
                'next_capitalized': '' if index == len(sentence) -1 else sentence[index+1][0].upper() == sentence[index+1][0],
                'last_capitalized': '' if index == 0 else sentence[index-1][0].upper() == sentence[index-1][0],
                'is_all_caps': word.upper() == word,
                'is_all_lower': word.lower() == word,
                'prefix-1-low': word[0].lower(),
                'prefix-1': word[0],
                'prefix-2': word[:2],
                'prefix-3': word[:3],
                'prefix-4': word[:4],
                'suffix-1': word[-1],
                'suffix-2': word[-2:],
                'suffix-3': word[-3:],
                'suffix-4': word[-4:],
                'prev_word': '' if index == 0 else sentence[index-1][0],
                'prev_prev_word': '' if index == 0 or index == 1 else sentence[index-2][0],
                'next_word': '' if index == len(sentence) - 1 else sentence[index + 1][0],
                'next_next_word': '' if index == len(sentence) - 1 or index == len(sentence) -2  else sentence[index + 2][0],
                #'prev_tag': '' if index == 0 else sentence[index-1][1],
                #'next_tag': '' if index == len(sentence)-1 else sentence[index+1][1],
                'has_hyphen': '-' in word,
                'is_numeric': word.isdigit(),
                'capitals_inside': word[1:].lower() != word[1:]
        }
        return features


def sent2features(sentence):
        return [word2features(sentence, i) for i in range(len(sentence))]

def analyze_meter(meter_model, string_line):
	#line = [('Ich', '+'), ('bin', '-'), ('ei', '+'), ('ne', '-'), ('mut', '+'), ('ter', '-')]
	#line = [('Ich', ' '), ('bin', ''), ('ei', ''), ('ne', ''), ('mut', ''), ('ter', '')]
	tokenized_line = tokenizer.tokenize(string_line)
	syllable_line = []
	for token in tokenized_line:
		hyphenated = dic.inserted(token)
		syllables = hyphenated.split('-')
		for syllable in syllables:
			syllable_line.append((syllable.strip(), ''))
	#print(syllable_line)

	line_features = sent2features(syllable_line)
	#print('Load Meter Model')
	clf = joblib.load(meter_model)
	#print('Predict Meter')
	pred = clf.predict([line_features])
	return pred

	
def get_meter(meter_model, string_line):
	#line = [('Ich', '+'), ('bin', '-'), ('ei', '+'), ('ne', '-'), ('mut', '+'), ('ter', '-')]
	#line = [('Ich', ' '), ('bin', ''), ('ei', ''), ('ne', ''), ('mut', ''), ('ter', '')]
	tokenized_line = tokenizer.tokenize(string_line)
	syllable_line = []
	for token in tokenized_line:
		hyphenated = dic.inserted(token)
		syllables = hyphenated.split('-')
		for syllable in syllables:
			syllable_line.append((syllable.strip(), ''))
	#print(syllable_line)

	line_features = sent2features(syllable_line)
	#print('Load Meter Model')
	#clf = joblib.load(meter_model)
	#print('Predict Meter')
	pred = meter_model.predict([line_features])
	return pred

def get_syllable_indices(string_line):
	words = []
	syllable_line = []
	w_s_mapping = {}
	tokenized_line = tokenizer.tokenize(string_line)
	t = 0
	s = 0
	for token in tokenized_line:
		#print(token)
		words.append((token, ''))
		hyphenated = dic.inserted(token)
		syllables = hyphenated.split('-')
		for syllable in syllables:
			#print(syllable)
			syllable_line.append((syllable.strip(), ''))
			w_s_mapping.setdefault(t, []).append(s)
			s+=1
		t+=1
	return words, syllable_line, w_s_mapping

def get_pos_label(label):
	if label.startswith('ADV'):
		return 'ADV'
	elif label.startswith('ADJ'):
		return 'ADJ'
	else:
		return label[:2]

def get_pos_meter_map(pos_line, meter_line, p_m_map):
	final_map = []
	p = 0
	for pos in pos_line:
		#print(pos)
		p_m = []
		m_indices = p_m_map[p]
		for m_index in m_indices:
			#print(m_index)
			p_m.append(meter_line[m_index])
		final_map.append((get_pos_label(pos), tuple(p_m)))
		p+=1
	return final_map


def get_pos_meter_mapping(pos_model, meter_model, line):
	words, syllable_line, mapping = get_syllable_indices(line)
	#print(words, syllable_line, mapping)

	sent_features = sent2features(words)
	#print(sent_features)
	syll_features = sent2features(syllable_line)
	#print(syll_features)

	pos = pos_model.predict([sent_features])[0]
	meter = meter_model.predict([syll_features])[0]
	#print(pos)
	#print(meter)

	mp = get_pos_meter_map(pos, meter, mapping)
	return mp

def get_pos_sequence(pos_model, string_line):
	tokenized_line = tokenizer.tokenize(string_line)
	tokenized_line = [(i, '') for i in tokenized_line]
	sent_features = sent2features(tokenized_line)
	pos = pos_model.predict([sent_features])[0]
	return pos

def get_versification(meter_line):
	label = None
	meter = ''.join(meter_line)
	meter = re.sub('\+', 'I', meter)
	meter = re.sub('\-', 'o', meter)
	#print(meter)
	iambicseptaplus = re.compile("oIoIoIoIoIoIoIo?")
	hexameter =       re.compile('Ioo?Ioo?Ioo?Ioo?IooIo$')
	alxiambichexa =   re.compile("oIoIoIoIoIoIo?$")
	iambicpenta =     re.compile("oIoIoIoIoIo?$")
	iambictetra =     re.compile("oIoIoIoIo?$")
	iambictri =       re.compile("oIoIoIo?$")
	iambicdi =        re.compile("oIoIo?$")
	iambic =          re.compile("oIoIo?")
	trochseptaplus =  re.compile('IoIoIoIoIoIoIo?')
	trochhexa =       re.compile('IoIoIoIoIoIo?$')
	trochpenta =      re.compile('IoIoIoIoIo?$')
	trochtetra =      re.compile('IoIoIoIo?$')
	trochtri =        re.compile('IoIoIo?$')
	trochdi =         re.compile('IoIo?$')
	troch =           re.compile('IoIo?')
	artemajor =       re.compile('oIooIooIooIo$')
	artemajorhalf =   re.compile('oIooIo$')
	zehnsilber =      re.compile('...I.....I$') 
	amphidi =         re.compile('oIooIo')
	amphitri =        re.compile('oIooIooIo')
	adontrochamphi =  re.compile('IooIo$')
	adoneusspond =    re.compile('IooII$')
	iambamphi =       re.compile('oIooI$')
	iambchol =        re.compile('IooI')
	anapaestdiplus =  re.compile('ooIooI')
	daktyldiplus =    re.compile('IooIoo')
	anapaestinit =    re.compile('ooI')
	daktylinit =      re.compile('Ioo')
	#alexandriner =    re.compile('oIoIoIoIoIoIo?$')
	#adoneus =        re.compile('IooIo$')

	verses = {'iambic.septa.plus':iambicseptaplus,\
		  'hexameter':hexameter,\
		  'alexandr.iambic.hexa':alxiambichexa,\
		  'iambic.penta':iambicpenta,\
		  'iambic.tetra':iambictetra,\
		  'iambic.tri':iambictri,\
		  'iambic.di':iambicdi,\
		  'troch.septa.plus':trochseptaplus,\
		  'troch.hexa':trochhexa,\
		  'troch.penta':trochpenta,\
		  'troch.tetra':trochtetra,\
		  'troch.tri':trochtri,\
		  'troch.di':trochdi,\
		  'arte_major':artemajor,\
		  'arte_major.half':artemajorhalf,\
		  'zehnsilber':zehnsilber,\
		  'adoneus.troch.amphi':adontrochamphi,\
		  'adoneus.spond':adoneusspond,\
		  'amphi.tri.plus':amphitri,\
		  'amphi.iamb':iambamphi,\
		  'amphi.di.plus':amphidi,\
		  'chol.iamb':iambchol,\
		  'iambic.mix':iambic,\
		  'troch.mix':troch,\
		  'anapaest.di.plus':anapaestdiplus,\
		  'daktyl.di.plus':daktyldiplus,\
		  'anapaest.init':anapaestinit,\
		  'daktyl.init':daktylinit}

	for label, pattern in verses.items():
		result = pattern.match(meter)
		if label == 'chol.iamb':
			result = pattern.search(meter)
		if result != None:
			return label
	else: return 'other'
	
def go_meter(meter_model_path, corpus_path):
	print('Loading Meter Model')
	meter_model = joblib.load(meter_model_path)
	print('Loading Corpus')
	corpus = json.load(open(corpus_path, 'r'))

	#get_pos_meter_mapping(pos_model, meter_model, line)

	pos_dict = {}

	all_versemeters = []
	counter = 0
	for idx, doc in corpus.items():
		counter += 1
		#print(counter)
		#if counter > 10000:
		#	break
		lines = doc['lines']
		for line in lines:
			meter = get_meter(meter_model, line)[0]
			verse = get_versification(meter)
			#print(counter, verse, meter)
			if verse != None:
				if verse == 'other':
					#print(counter, verse,"".join(meter), line)
					#all_versemeters.append("".join(meter)[:8])
					all_versemeters.append('other')
				else:
					all_versemeters.append(verse)

	cnt = Counter(all_versemeters)
	print(len(all_versemeters), cnt)
	plot_versemeter(cnt)

def go_pos_meter(pos_model_path, meter_model_path, corpus_path):
	print('Loading POS Model')
	pos_model = joblib.load(pos_model_path)
	print('Loading Meter Model')
	meter_model = joblib.load(meter_model_path)
	print('Loading Corpus')
	corpus = json.load(open(corpus_path, 'r'))

	#get_pos_meter_mapping(pos_model, meter_model, line)

	pos_dict = {}

	all_versemeters = []
	counter = 0
	for idx, doc in corpus.items():
		counter += 1
		#print(counter)
		if counter > 2000:
			break
		author = doc['author']
		lines = doc['lines']
		#year = doc['year']
		print(counter)
		#print(os.system('clear'))
		for line in lines:
			meter = get_meter(meter_model, line)[0]
			pos = get_pos_sequence(pos_model, line)
			#verse = get_versification(meter)
			#vsplit = verse.split('.')
			#vinit = vsplit[0]
			#print(counter, verse, meter)
			#if verse != None:
			pm = get_pos_meter_mapping(pos_model, meter_model, line)
			for p, m in pm:
				cnt = pos_dict.setdefault(p, Counter())
				cnt[m] += 1
			#cnt[vinit] +=1

	#cnt = Counter(all_versemeters)
	#print(len(all_versemeters), cnt)
	#plot_versemeter(cnt)
	#print(pos_dict)
	c = 0
	for item in sorted(pos_dict.items()):
		c+=1
		print()
		print(c, len(pos_dict.keys()), item)
	plot_authors(pos_dict)
	return pos_dict

def go_author_meter(pos_model_path, meter_model_path, corpus_path):
	print('Loading POS Model')
	pos_model = joblib.load(pos_model_path)
	print('Loading Meter Model')
	meter_model = joblib.load(meter_model_path)
	print('Loading Corpus')
	corpus = json.load(open(corpus_path, 'r'))

	#get_pos_meter_mapping(pos_model, meter_model, line)

	pos_dict = {}

	all_versemeters = []
	counter = 0
	for idx, doc in corpus.items():
		counter += 1
		#print(counter)
		#if counter > 1000:
		#	break
		author = doc['author']
		lines = doc['lines']
		#year = doc['year']
		print(counter)
		#print(os.system('clear'))
		for line in lines:
			meter = get_meter(meter_model, line)[0]
			pos = get_pos_sequence(pos_model, line)
			verse = get_versification(meter)
			vsplit = verse.split('.')
			vinit = vsplit[0]
			#print(counter, verse, meter)
			if verse != None:
				cnt = pos_dict.setdefault(author, Counter())
				#cnt["_".join(pos[:3])] += 1
				cnt[vinit] +=1

	#cnt = Counter(all_versemeters)
	#print(len(all_versemeters), cnt)
	#plot_versemeter(cnt)
	#print(pos_dict)
	c = 0
	for item in sorted(pos_dict.items()):
		c+=1
		print()
		print(c, len(pos_dict.keys()), item)
	plot_authors(pos_dict)
	return pos_dict

def plot_authors(author_verse_dict):
	c = 0
	for author, counter in author_verse_dict.items():
		c+=1
		print(c, author, counter)
		plot_author(author, counter)

def plot_author(author, cnt):
	counter = cnt.most_common()
	counter.reverse()
	labels, values = zip(*counter)

	l = sum(values)
	indexes = np.arange(len(labels))
	width = .8

	plt.figure(figsize=(10,10))
	plt.barh(indexes, values, width, color='black')
	#plt.xticks(indexes + width * 0.5, labels, rotation=-45)
	plt.yticks(indexes, labels, rotation='horizontal', fontsize=22, color='red')
	for i, v in enumerate(values):
		plt.text(v + v*.02, i -.05, str(str(v) + '  |  ' + str(round(v/l,3))), color='black', fontsize=18, fontweight='normal')
	plt.margins(0.2)
	# Tweak spacing to prevent clipping of tick-labels
	plt.subplots_adjust(left=0.2)
	plt.title(author + '; ' + str(sum(values)) + ' lines', fontsize='28')
	f = open('versemeter/' + '_'.join(author.split(' ')) + '_versemeter.png', 'wb')
	plt.savefig(f)
	plt.close()
	f.close()

def plot_versemeter(cnt):
	counter = cnt.most_common()
	counter.reverse()
	labels, values = zip(*counter)

	l = sum(values)
	indexes = np.arange(len(labels))
	width = .8

	plt.figure(figsize=(10,10))
	plt.barh(indexes, values, width)
	#plt.xticks(indexes + width * 0.5, labels, rotation=-45)
	plt.yticks(indexes, labels, rotation='horizontal', fontsize=10)
	for i, v in enumerate(values):
		plt.text(v + 200, i -.25, str(str(v) + ' : ' + str(round(v/l,3))), color='black', fontsize=6, fontweight='normal')
	plt.margins(0.05)
	# Tweak spacing to prevent clipping of tick-labels
	plt.subplots_adjust(left=0.2)
	plt.title('Poetic Meter of 1.6M lines')
	plt.savefig('versemeter.finer.png')

def has_enjambment(string_line):
	for punc in string.punctuation:
		if string_line.strip().endswith(punc):
			return False
	else: return True

def go_enjambment_pos(pos_model_path, meter_model_path, corpus_path):
	print('Loading POS Model')
	pos_model = joblib.load(pos_model_path)
	print('Loading Meter Model')
	meter_model = joblib.load(meter_model_path)
	print('Loading Corpus')
	corpus = json.load(open(corpus_path, 'r'))

	#get_pos_meter_mapping(pos_model, meter_model, line)

	pos_dict = {}

	all_versemeters = []
	counter = 0
	true_enj = []
	false_enj = []
	for idx, doc in corpus.items():
		counter += 1
		#print(counter)
		#if counter > 100000:
		#	break
		#author = doc['author']
		lines = doc['lines']
		#year = doc['year']
		print(counter)
		#print(os.system('clear'))
		print()
		pos_seqs = []
		enj_seqs = []
		for line in lines:
			#if len(line.strip()) < 1:
			#	continue
			#meter = get_meter(meter_model, line)[0]
			pos = get_pos_sequence(pos_model, line)
			pos = [get_pos_label(p) for p in pos]
			pos_seqs.append(pos)
			#verse = get_versification(meter)
			#vsplit = verse.split('.')
			#vinit = vsplit[0]
			enj = has_enjambment(line)
			enj_seqs.append(enj)
			#print(enj, line)
		#print(pos_seqs)
		#print(enj_seqs)
		for i, enj in enumerate(enj_seqs):
			#print(i)
			#print(enj_seqs[i])
			#print(pos_seqs[i])
			try:
				if i < len(enj_seqs)-1:
					if enj == True:
						transt = ".".join([pos_seqs[i][-1],pos_seqs[i+1][0]])
						true_enj.append(transt)
					elif enj == False:
						transf = ".".join([pos_seqs[i][-1],pos_seqs[i+1][0]])
						false_enj.append(transf)
			except IndexError:
				continue
	tru_e = Counter(true_enj)
	fal_e = Counter(false_enj)

	print('True Enjamb: ', tru_e)
	print()
	print('False Enjamb: ', fal_e)

def go_enjambment_meter(pos_model_path, meter_model_path, corpus_path):
	print('Loading POS Model')
	pos_model = joblib.load(pos_model_path)
	print('Loading Meter Model')
	meter_model = joblib.load(meter_model_path)
	print('Loading Corpus')
	corpus = json.load(open(corpus_path, 'r'))

	#get_pos_meter_mapping(pos_model, meter_model, line)

	pos_dict = {}

	all_versemeters = []
	counter = 0
	meter_enj = {}
	for idx, doc in corpus.items():
		counter += 1
		#print(counter)
		#if counter > 1000:
		#	break
		#author = doc['author']
		lines = doc['lines']
		#year = doc['year']
		print(counter)
		#print(os.system('clear'))
		print()
		for line in lines:
			#if len(line.strip()) < 1:
			#	continue
			meter = get_meter(meter_model, line)[0]
			#pos = get_pos_sequence(pos_model, line)
			#pos = [get_pos_label(p) for p in pos]
			#pos_seqs.append(pos)
			verse = get_versification(meter)
			#vsplit = verse.split('.')
			#vinit = vsplit[0]
			enj = has_enjambment(line)
			cnt = meter_enj.setdefault(verse, Counter())
			cnt[enj] += 1
			
	print(meter_enj)
	m_e = []
	for v, c in meter_enj.items():
		t = float(c[True])
		f = float(c[False])
		m_e.append((round(t/(t+f),3), v, c))
	m = sorted(m_e)
	m.reverse()
	for i in m:
		print(i)

def go_pos_context(pos_model_path, meter_model_path, corpus_path):
	from nltk import bigrams
	#line = 'Laut zerspringt der Weiherspiegel.'
	#print(line)

	pos_model = joblib.load(pos_model_path)
	meter_model = joblib.load(meter_model_path)
	corpus = json.load(open(corpus_path, 'r'))

	#get_pos_meter_mapping(pos_model, meter_model, line)

	pos_dict = {}

	counter = 0
	for idx, doc in corpus.items():
		counter += 1
		if counter > 20000:
			break
		lines = doc['lines']
		for line in lines:
			mp = get_pos_meter_mapping(pos_model, meter_model, line)
			for tuple1, tuple2 in bigrams(mp):
				pos1 = tuple1[0]
				meter1 = tuple1[1]
				pos2 = tuple2[0]
				meter2 = tuple2[1]
				#print(pos, meter)
				cnt = pos_dict.setdefault("_".join([pos1,pos2]), Counter())
				cnt[meter2] += 1
				

	#print(pos_dict)
	ranking = []

	for pos, contours in pos_dict.items():
		ps = pos.split('_')
		pos1 = ps[0]
		pos2 = ps[1]
		print(pos1, pos2, contours)

		plus1 = 1
		minus1 = 1
		plus2 = 1
		minus2 = 1
		amphi = 1
		dibrach = 1
		spondee = 1
		for c in contours:
			if len(c) == 1:
				if pos2 == 'VM':
					print(pos1, pos2, c, contours[c])
					if c[0] == '+':
						plus1+= float(contours[c])
					elif c[0] == '-':
						minus1+= float(contours[c])
			#if len(c) > 1:
			#	print(pos, c, contours[c])
			#	for prom in c:
			#		if prom == '+':
			#			plus2+= float(contours[c]/len(c))
			#		if prom == '-':
			#			minus2+= float(contours[c]/len(c))
				
			#if len(c) == 2:
			#	print(pos, c, c[0], contours[c])
			#	if c[0] == '+' and c[1] == '-':
			#		plus2 = float(contours[c])
			#	elif c[0] == '-' and c[1] == '+':
			#		minus2 = float(contours[c])
			#	elif c[0] == '-' and c[1] == '-':
			#		dibrach = float(contours[c])
			#	elif c[0] == '+' and c[1] == '+':
			#		spondee = float(contours[c])
			#if len(c) == 3:
			#	print(pos, c, c[0], contours[c])
			#	if c[0] == '+' and c[1] == '-' and c[2] == '-':
			#		plus = float(contours[c])
			#	elif c[0] == '-' and c[1] == '-' and c[2] == '+':
			#		minus = float(contours[c])
			#	elif c[0] == '-' and c[1] == '+' and c[2] == '-':
			#		amphi = float(contours[c])
					
		if pos2 == 'VM':
			plus = plus1 + plus2
			minus = minus1 + minus2
			print("_".join([pos1,pos2]), round(plus, 2), round(minus, 2))
			ranking.append((round(plus/minus,2), "_".join([pos1,pos2])))

					
		#einsilber = plus1/minus1
		#zweisilber = (plus2+minus2+spondee)/dibrach
		#print(pos, round(einsilber, 2), round(zweisilber, 2))
		#ranking.append((round(einsilber,2), pos))
		
	s = sorted(ranking)
	s.reverse()
	print(s)


	#meter = analyze_meter(meter_model_path, sentence)[0]

	#print(meter)

def go(pos_model_path, meter_model_path, corpus_path):
	#line = 'Laut zerspringt der Weiherspiegel.'
	#print(line)

	pos_model = joblib.load(pos_model_path)
	meter_model = joblib.load(meter_model_path)
	corpus = json.load(open(corpus_path, 'r'))

	#get_pos_meter_mapping(pos_model, meter_model, line)

	pos_dict = {}

	counter = 0
	for idx, doc in corpus.items():
		counter += 1
		if counter > 1000:
			break
		lines = doc['lines']
		for line in lines:
			mp = get_pos_meter_mapping(pos_model, meter_model, line)
			for tuples in mp:
				pos = tuples[0]
				meter = tuples[1]
				#print(pos, meter)
				cnt = pos_dict.setdefault(pos, Counter())
				cnt[meter] += 1
				

	#print(pos_dict)
	ranking = []

	i = 0

	for pos, contours in pos_dict.items():
		plus1 = 1
		minus1 = 1
		plus2 = 1
		minus2 = 1
		amphi = 1
		dibrach = 1
		spondee = 1
		for c in contours:
			if len(c) == 1:
				print(pos, c, contours[c])
				if c[0] == '+':
					plus1+= float(contours[c])
				elif c[0] == '-':
					minus1+= float(contours[c])
			#if len(c) > 1:
			#	print(pos, c, contours[c])
			#	for prom in c:
			#		if prom == '+':
			#			plus2+= float(contours[c]/len(c))
			#		if prom == '-':
			#			minus2+= float(contours[c]/len(c))
				
			#if len(c) == 2:
			#	print(pos, c, c[0], contours[c])
			#	if c[0] == '+' and c[1] == '-':
			#		plus2 = float(contours[c])
			#	elif c[0] == '-' and c[1] == '+':
			#		minus2 = float(contours[c])
			#	elif c[0] == '-' and c[1] == '-':
			#		dibrach = float(contours[c])
			#	elif c[0] == '+' and c[1] == '+':
			#		spondee = float(contours[c])
			#if len(c) == 3:
			#	print(pos, c, c[0], contours[c])
			#	if c[0] == '+' and c[1] == '-' and c[2] == '-':
			#		plus = float(contours[c])
			#	elif c[0] == '-' and c[1] == '-' and c[2] == '+':
			#		minus = float(contours[c])
			#	elif c[0] == '-' and c[1] == '+' and c[2] == '-':
			#		amphi = float(contours[c])
					
		
		plus = plus1 + plus2
		minus = minus1 + minus2
		print(pos, round(plus, 2), round(minus, 2))
		ranking.append((round(plus/minus,2), pos))

					
		#einsilber = plus1/minus1
		#zweisilber = (plus2+minus2+spondee)/dibrach
		#print(pos, round(einsilber, 2), round(zweisilber, 2))
		#ranking.append((round(einsilber,2), pos))
		
	s = sorted(ranking)
	s.reverse()
	print(s)



go_pos_context(sys.argv[1], sys.argv[2], sys.argv[3])
#go()
#go_meter(sys.argv[1], sys.argv[2])
#go_author_meter(sys.argv[1], sys.argv[2], sys.argv[3])
#go_enjambment_meter(sys.argv[1], sys.argv[2], sys.argv[3])
#go_pos_meter(sys.argv[1], sys.argv[2], sys.argv[3])
