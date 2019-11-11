#!/usr/bin/env python 
#-*- coding: utf-8 -*-
import sys
import re
import json
from collections import defaultdict
import operator

import joblib

from inout.dta.corpus import Corpus
from inout.dta.poem import Poem

from nltk.tokenize import RegexpTokenizer
import pyphen
from nltk import bigrams as bi

import random
import sklearn_crfsuite
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn_crfsuite import metrics
from pandas_ml import ConfusionMatrix

dic = pyphen.Pyphen(lang='de_DE')
tokenizer = RegexpTokenizer(r'\w+')
prob_dict = defaultdict(int)

forbidden_words = ['jn', '—', "b'<l'"]
xmlns = 'xmlns'


def cleanup_routine(word):
	#print(word)
	endword = str(word)
	endword = re.sub('<[^>]*>', '', endword)
	#print('tags', endword)
	endword = re.sub('/', '', endword)
	#print('slash', endword)
	endword = re.sub('\\,', '', endword)
	#print('comma', endword)
	endword = re.sub('\\.', '', endword)
	#print('dot', endword)
	endword = re.sub('\\:', '', endword)
	#print('colon', endword)
	endword = re.sub('\\;', '', endword)
	#print('semi', endword)
	endword = re.sub('\\?', '', endword)
	#print('question', endword)
	endword = re.sub('\\!', '', endword)
	#print('exlamation', endword)
	endword = re.sub('„', '', endword)
	#print('apostophe', endword)
	endword = re.sub('ſ', 's', endword)
	endword = re.sub('ey', 'ei', endword)
	endword = re.sub('ff', 'f', endword)
	endword = re.sub('th$', 't$', endword)
	endword = re.sub('jhr', 'ihr', endword)
	endword = re.sub('jst', 'ist', endword)
	endword = re.sub('kan', 'kann', endword)
	endword = re.sub('uͤr', 'ür', endword)
	endword = re.sub('uͤl', 'ül', endword)
	endword = re.sub('uͤh', 'üh', endword)
	endword = re.sub('uͤss', 'üss', endword)
	endword = re.sub('uͤb', 'üb', endword)
	endword = re.sub('uͤt', 'üt', endword)
	endword = re.sub('uͤg', 'üg', endword)
	endword = re.sub('uͤc', 'üc', endword)
	endword = re.sub('uͤn', 'ün', endword)
	endword = re.sub('uͤs', 'üs', endword)
	endword = re.sub('aͤu', 'au', endword)
	endword = re.sub('aͤr', 'är', endword)
	endword = re.sub('aͤf', 'äf', endword)
	endword = re.sub('aͤn', 'än', endword)
	endword = re.sub('aͤl', 'äl', endword)
	endword = re.sub('aͤc', 'äc', endword)
	endword = re.sub('aͤm', 'äm', endword)
	endword = re.sub('aͤh', 'äh', endword)
	endword = re.sub('oͤß', 'öß', endword)
	endword = re.sub('oͤs', 'ös', endword)
	endword = re.sub('oͤc', 'öc', endword)
	endword = re.sub('oͤp', 'öp', endword)
	endword = re.sub('oͤh', 'öh', endword)
	endword = re.sub('qv', 'qu', endword)
	endword = re.sub('^th', '^t', endword)
	endword = re.sub('\\^', '', endword)
	endword = re.sub('jch', 'ich', endword)
	endword = re.sub('\\$$', '', endword)
	#print('output', endword)
	if endword in forbidden_words or xmlns in endword:
		return None
	else:
		return str(endword.encode('utf-8').decode('utf-8'))

def get_rhythm_annotation(corpuspath):
	line_annotation = []
	c = Corpus(corpuspath)
	poems = c.get_poems()
	p = 0
	l = 0
	for poem in poems:
		p +=1
		#if p > 10:
		#	continue
		#print()
		print(poem.get_author(),poem.get_title(),poem.get_year())
		stanzas = poem.get_stanzas()
		tokenized_lines = []
		s = 0
		for stanza in stanzas:
			s +=1 
			if s > len(stanzas)/2.:
				continue
			#print('STANZA')
			lines = stanza.get_line_objects()
			for line in lines:
				l += 1
				#print(p, len(poems), l)
				#print(line)
				tokenized_line = tokenizer.tokenize(line.get_text())
				#print(tokenized_line)
				syllable_line = []
				for token in tokenized_line:
					hyphenated = dic.inserted(token)
					syllables = hyphenated.split('-')
					for syllable in syllables:
						syllable_line.append(cleanup_routine(syllable))
				#print(syllable_line)
				rhythm1 = line.get_rhythm()
				rhythm = str(re.sub('\|', ':', str(rhythm1)))
				rhythm = str(re.sub('\(', '', str(rhythm)))
				rhythm = str(re.sub('\)', '', str(rhythm)))
				rhythmsymbols = str(re.sub(':', '', str(rhythm)))
				rhythm = [m for m in rhythm]
				caesuras = []
				rhythmcaesuras = []
				for a, b in bi(rhythm):
					if a == ':':
						continue
					elif b == ':':
						caesuras.append('c')
						rhythmcaesuras.append(a+b)
					else:
						caesuras.append('0')
						rhythmcaesuras.append(a)
				if rhythm1 is not None and len(syllable_line) == len(rhythmsymbols):
					#print(syllable_line, meter, len(syllable_line), len(meter))
					tups = list(zip(syllable_line, rhythmcaesuras))
					#print(tups)
					line_annotation.append(tups)
	#print(len(line_annotation))
	#lines = [zip(i) for i in line_annotation]
	
	return line_annotation

def get_meter_annotation(corpuspath):
	line_annotation = []
	c = Corpus(corpuspath)
	poems = c.get_poems()
	p = 0
	l = 0
	for poem in poems:
		p +=1
		#if p > 10:
		#	continue
		#print()
		print(poem.get_author(),poem.get_title(),poem.get_year())
		stanzas = poem.get_stanzas()
		tokenized_lines = []
		s = 0
		for stanza in stanzas:
			s +=1 
			if s > len(stanzas)/2.:
				continue
			#print('STANZA')
			lines = stanza.get_line_objects()
			for line in lines:
				l += 1
				#print(p, len(poems), l)
				#print(line)
				tokenized_line = tokenizer.tokenize(line.get_text())
				#tokenized_lines = line.get_text()
				#tokenized_line = tokenized_lines.split()
				print(tokenized_line)
				syllable_line = []
				for token in tokenized_line:
					hyphenated = dic.inserted(token)
					syllables = hyphenated.split('-')
					for syllable in syllables:
						syllable_line.append(syllable)
				#print(syllable_line)
				meter1 = line.get_meter()
				meter = [m for m in re.sub('\|', '', str(meter1))]
				bimeter = bi([m for m in re.sub('\|', '.', str(meter1))])
				feetmeter = []
				feet = []	
				for a, b in bimeter:
					if a == '.':
						continue
					elif b == '.':
						feetmeter.append(a+b)
						feet.append(':')
					else:
						feetmeter.append(a)
						feet.append('x')
				
				if meter1 is not None and len(syllable_line) == len(meter):
					#print(syllable_line, meter, len(syllable_line), len(meter))
					tups = list(zip(syllable_line, meter))
					#tups = list(zip(syllable_line, feetmeter))
					#print(tups)
					line_annotation.append(tups)
	#print(len(line_annotation))
	#lines = [zip(i) for i in line_annotation]
	
	return line_annotation
	

def word2features_test(sentence, index):
	word = sentence[index][0]
	postag = sentence[index][1]
	features = {
	# übernommen vom DecisionTreeClassifier
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

def word2features(sentence, index):
	word = sentence[index][0]
	postag = sentence[index][1]
	features = {
	# übernommen vom DecisionTreeClassifier
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
		'prev_tag': '' if index == 0 else sentence[index-1][1],
		'next_tag': '' if index == len(sentence)-1 else sentence[index+1][1],
		'has_hyphen': '-' in word,
		'is_numeric': word.isdigit(),
		'capitals_inside': word[1:].lower() != word[1:]
	}
	return features

def sent2features(sentence):
	return [word2features(sentence, i) for i in range(len(sentence))]

def sent2labels(sentence):
	return [w[1] for w in sentence]

def go():
	lines = get_rhythm_annotation(sys.argv[1])
	#lines = get_meter_annotation(sys.argv[1])
	#print(lines)
	X = [sent2features(sentence) for sentence in lines]
	y = [sent2labels(sentence) for sentence in lines]
	#X, y = zip(*lines)
	#print(y)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

	classifier = sklearn_crfsuite.CRF()

	classifier.fit(X_train, y_train)
	#print(list(zip(X_test, y_test)))
	print('Train Size', len(X_train))
	print('Test Size', len(X_test))
	y_pred = classifier.predict(X_test)
	flat_y_test = [item for sublist in y_test for item in sublist]
	flat_y_pred = [item for sublist in y_pred for item in sublist]
	print('Acc ', classifier.score(X_test, y_test))
	cm = ConfusionMatrix(flat_y_test, flat_y_pred)
	print(cm)

	outlines = lines[-3:]
	Xoutlines = [sent2features(line) for line in outlines]
	pred = classifier.predict(Xoutlines)
	print(list(zip(outlines, pred)))
	labels = list(classifier.classes_)
	#labels.remove(' ')
	sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
	print(metrics.flat_classification_report(
   	y_test, y_pred, labels=sorted_labels, digits=3))
	#outfile = open('footmeter.model.joblib', 'w')
	joblib.dump(classifier, 'caesura.rhythm.model.joblib')

go()
#get_meter_annotation(sys.argv[1])
'''
classifier = sklearn_crfsuite.CRF()
print('loading corpus')
with open ('dta_komplett_tcf-full-ohne_lyrik-1000000sents', 'rb') as f:
	train_sents = pickle.load(f)

with open ('dta_komplett_tcf-full-lyrik-1000000sents', 'rb') as f:
	sents = pickle.load(f)

print('preparing corpus')
random.shuffle(sents)
random.shuffle(train_sents)

cutoff = int(0.8 * len(sents))
#train_sents = sents[:cutoff]
test_sents = sents[cutoff:]

Features = [sent2features(s) for s in train_sents]
tags = [sent2labels(s) for s in train_sents]

print('training')
classifier.fit(Features, tags)

print('testing')
Features_test = [sent2features(s) for s in test_sents]
tags_test = [sent2labels(s) for s in test_sents]
print('Accuracy: ', classifier.score(Features_test, tags_test))
'''
