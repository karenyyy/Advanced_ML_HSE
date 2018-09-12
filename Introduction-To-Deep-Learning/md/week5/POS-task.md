
__This seminar:__ after you're done coding your own recurrent cells, it's time you learn how to train recurrent networks easily with Keras. We'll also learn some tricks on how to use keras layers and model. We also want you to note that this is a non-graded assignment, meaning you are not required to pass it for a certificate.

Enough beatin' around the bush, let's get to the task!

## Part Of Speech Tagging

<img src=https://i.stack.imgur.com/6pdIT.png width=320>

Unlike our previous experience with language modelling, this time around we learn the mapping between two different kinds of elements.

This setting is common for a range of useful problems:
* Speech Recognition - processing human voice into text
* Part Of Speech Tagging - for morphology-aware search and as an auxuliary task for most NLP problems
* Named Entity Recognition - for chat bots and web crawlers
* Protein structure prediction - for bioinformatics

Our current guest is part-of-speech tagging. As the name suggests, it's all about converting a sequence of words into a sequence of part-of-speech tags. We'll use a reduced tag set for simplicity:

### POS-tags
- ADJ - adjective (new, good, high, ...)
- ADP - adposition	(on, of, at, ...)
- ADV - adverb	(really, already, still, ...)
- CONJ	- conjunction	(and, or, but, ...)
- DET - determiner, article	(the, a, some, ...)
- NOUN	- noun	(year, home, costs, ...)
- NUM - numeral	(twenty-four, fourth, 1991, ...)
- PRT -	particle (at, on, out, ...)
- PRON - pronoun (he, their, her, ...)
- VERB - verb (is, say, told, ...)
- .	- punctuation marks	(. , ;)
- X	- other	(ersatz, esprit, dunno, ...)


```python
import nltk
import sys
import numpy as np
nltk.download('brown')
nltk.download('universal_tagset')
data = nltk.corpus.brown.tagged_sents(tagset='universal')
all_tags = ['#EOS#','#UNK#','ADV', 'NOUN', 'ADP', 'PRON', 'DET', '.', 'PRT', 'VERB', 'X', 'NUM', 'CONJ', 'ADJ']

data = np.array([ [(word.lower(),tag) for word,tag in sentence] for sentence in data ])
```

    [nltk_data] Downloading package brown to /home/jovyan/nltk_data...
    [nltk_data]   Unzipping corpora/brown.zip.
    [nltk_data] Downloading package universal_tagset to
    [nltk_data]     /home/jovyan/nltk_data...
    [nltk_data]   Unzipping taggers/universal_tagset.zip.



```python
data
```




    array([ [('the', 'DET'), ('fulton', 'NOUN'), ('county', 'NOUN'), ('grand', 'ADJ'), ('jury', 'NOUN'), ('said', 'VERB'), ('friday', 'NOUN'), ('an', 'DET'), ('investigation', 'NOUN'), ('of', 'ADP'), ("atlanta's", 'NOUN'), ('recent', 'ADJ'), ('primary', 'NOUN'), ('election', 'NOUN'), ('produced', 'VERB'), ('``', '.'), ('no', 'DET'), ('evidence', 'NOUN'), ("''", '.'), ('that', 'ADP'), ('any', 'DET'), ('irregularities', 'NOUN'), ('took', 'VERB'), ('place', 'NOUN'), ('.', '.')],
           [('the', 'DET'), ('jury', 'NOUN'), ('further', 'ADV'), ('said', 'VERB'), ('in', 'ADP'), ('term-end', 'NOUN'), ('presentments', 'NOUN'), ('that', 'ADP'), ('the', 'DET'), ('city', 'NOUN'), ('executive', 'ADJ'), ('committee', 'NOUN'), (',', '.'), ('which', 'DET'), ('had', 'VERB'), ('over-all', 'ADJ'), ('charge', 'NOUN'), ('of', 'ADP'), ('the', 'DET'), ('election', 'NOUN'), (',', '.'), ('``', '.'), ('deserves', 'VERB'), ('the', 'DET'), ('praise', 'NOUN'), ('and', 'CONJ'), ('thanks', 'NOUN'), ('of', 'ADP'), ('the', 'DET'), ('city', 'NOUN'), ('of', 'ADP'), ('atlanta', 'NOUN'), ("''", '.'), ('for', 'ADP'), ('the', 'DET'), ('manner', 'NOUN'), ('in', 'ADP'), ('which', 'DET'), ('the', 'DET'), ('election', 'NOUN'), ('was', 'VERB'), ('conducted', 'VERB'), ('.', '.')],
           [('the', 'DET'), ('september-october', 'NOUN'), ('term', 'NOUN'), ('jury', 'NOUN'), ('had', 'VERB'), ('been', 'VERB'), ('charged', 'VERB'), ('by', 'ADP'), ('fulton', 'NOUN'), ('superior', 'ADJ'), ('court', 'NOUN'), ('judge', 'NOUN'), ('durwood', 'NOUN'), ('pye', 'NOUN'), ('to', 'PRT'), ('investigate', 'VERB'), ('reports', 'NOUN'), ('of', 'ADP'), ('possible', 'ADJ'), ('``', '.'), ('irregularities', 'NOUN'), ("''", '.'), ('in', 'ADP'), ('the', 'DET'), ('hard-fought', 'ADJ'), ('primary', 'NOUN'), ('which', 'DET'), ('was', 'VERB'), ('won', 'VERB'), ('by', 'ADP'), ('mayor-nominate', 'NOUN'), ('ivan', 'NOUN'), ('allen', 'NOUN'), ('jr.', 'NOUN'), ('.', '.')],
           ...,
           [('the', 'DET'), ('doors', 'NOUN'), ('of', 'ADP'), ('the', 'DET'), ('d', 'NOUN'), ('train', 'NOUN'), ('slid', 'VERB'), ('shut', 'VERB'), (',', '.'), ('and', 'CONJ'), ('as', 'ADP'), ('i', 'PRON'), ('dropped', 'VERB'), ('into', 'ADP'), ('a', 'DET'), ('seat', 'NOUN'), ('and', 'CONJ'), (',', '.'), ('exhaling', 'VERB'), (',', '.'), ('looked', 'VERB'), ('up', 'PRT'), ('across', 'ADP'), ('the', 'DET'), ('aisle', 'NOUN'), (',', '.'), ('the', 'DET'), ('whole', 'ADJ'), ('aviary', 'NOUN'), ('in', 'ADP'), ('my', 'DET'), ('head', 'NOUN'), ('burst', 'VERB'), ('into', 'ADP'), ('song', 'NOUN'), ('.', '.')],
           [('she', 'PRON'), ('was', 'VERB'), ('a', 'DET'), ('living', 'VERB'), ('doll', 'NOUN'), ('and', 'CONJ'), ('no', 'DET'), ('mistake', 'NOUN'), ('--', '.'), ('the', 'DET'), ('blue-black', 'ADJ'), ('bang', 'NOUN'), (',', '.'), ('the', 'DET'), ('wide', 'ADJ'), ('cheekbones', 'NOUN'), (',', '.'), ('olive-flushed', 'ADJ'), (',', '.'), ('that', 'PRON'), ('betrayed', 'VERB'), ('the', 'DET'), ('cherokee', 'NOUN'), ('strain', 'NOUN'), ('in', 'ADP'), ('her', 'DET'), ('midwestern', 'ADJ'), ('lineage', 'NOUN'), (',', '.'), ('and', 'CONJ'), ('the', 'DET'), ('mouth', 'NOUN'), ('whose', 'DET'), ('only', 'ADJ'), ('fault', 'NOUN'), (',', '.'), ('in', 'ADP'), ('the', 'DET'), ("novelist's", 'NOUN'), ('carping', 'VERB'), ('phrase', 'NOUN'), (',', '.'), ('was', 'VERB'), ('that', 'ADP'), ('the', 'DET'), ('lower', 'ADJ'), ('lip', 'NOUN'), ('was', 'VERB'), ('a', 'DET'), ('trifle', 'NOUN'), ('too', 'ADV'), ('voluptuous', 'ADJ'), ('.', '.')],
           [('from', 'ADP'), ('what', 'DET'), ('i', 'PRON'), ('was', 'VERB'), ('able', 'ADJ'), ('to', 'ADP'), ('gauge', 'NOUN'), ('in', 'ADP'), ('a', 'DET'), ('swift', 'ADJ'), (',', '.'), ('greedy', 'ADJ'), ('glance', 'NOUN'), (',', '.'), ('the', 'DET'), ('figure', 'NOUN'), ('inside', 'ADP'), ('the', 'DET'), ('coral-colored', 'ADJ'), ('boucle', 'NOUN'), ('dress', 'NOUN'), ('was', 'VERB'), ('stupefying', 'VERB'), ('.', '.')]], dtype=object)




```python
from sklearn.model_selection import train_test_split
train_data,test_data = train_test_split(data,test_size=0.25,random_state=42)
```


```python
train_data.shape, test_data.shape
```




    ((43005,), (14335,))



### Building vocabularies

Just like before, we have to build a mapping from tokens to integer ids. This time around, our model operates on a word level, processing one word per RNN step. This means we'll have to deal with far larger vocabulary.

Luckily for us, we only receive those words as input i.e. we don't have to predict them. This means we can have a large vocabulary for free by using word embeddings.


```python
from collections import Counter
word_counts = Counter()
for sentence in data:
    words,tags = zip(*sentence)
    word_counts.update(words)

all_words = ['#EOS#','#UNK#']+list(list(zip(*word_counts.most_common(10000)))[0])

#let's measure what fraction of data words are in the dictionary
print("Coverage = %.5f"%(float(sum(word_counts[w] for w in all_words)) / sum(word_counts.values())))
```

    Coverage = 0.92876



```python
word_counts
```




    Counter({'the': 69971,
             'fulton': 17,
             'county': 155,
             'grand': 48,
             'jury': 67,
             'said': 1961,
             'friday': 60,
             'an': 3740,
             'investigation': 51,
             'of': 36412,
             "atlanta's": 4,
             'recent': 179,
             'primary': 96,
             'election': 77,
             'produced': 90,
             '``': 8837,
             'no': 2139,
             'evidence': 204,
             "''": 8789,
             'that': 10594,
             'any': 1344,
             'irregularities': 8,
             'took': 426,
             'place': 570,
             '.': 49346,
             'further': 218,
             'in': 21337,
             'term-end': 1,
             'presentments': 1,
             'city': 393,
             'executive': 55,
             'committee': 168,
             ',': 58334,
             'which': 3561,
             'had': 5133,
             'over-all': 35,
             'charge': 122,
             'deserves': 16,
             'praise': 17,
             'and': 28853,
             'thanks': 37,
             'atlanta': 35,
             'for': 9489,
             'manner': 124,
             'was': 9815,
             'conducted': 55,
             'september-october': 1,
             'term': 79,
             'been': 2472,
             'charged': 57,
             'by': 5306,
             'superior': 46,
             'court': 230,
             'judge': 77,
             'durwood': 1,
             'pye': 1,
             'to': 26158,
             'investigate': 11,
             'reports': 84,
             'possible': 374,
             'hard-fought': 2,
             'won': 68,
             'mayor-nominate': 1,
             'ivan': 4,
             'allen': 20,
             'jr.': 75,
             'only': 1748,
             'a': 23195,
             'relative': 46,
             'handful': 13,
             'such': 1303,
             'received': 163,
             'considering': 47,
             'widespread': 30,
             'interest': 330,
             'number': 472,
             'voters': 20,
             'size': 138,
             'this': 5145,
             'it': 8760,
             'did': 1044,
             'find': 400,
             'many': 1030,
             "georgia's": 9,
             'registration': 23,
             'laws': 88,
             'are': 4394,
             'outmoded': 4,
             'or': 4206,
             'inadequate': 32,
             'often': 369,
             'ambiguous': 22,
             'recommended': 46,
             'legislators': 20,
             'act': 283,
             'have': 3942,
             'these': 1573,
             'studied': 79,
             'revised': 16,
             'end': 409,
             'modernizing': 4,
             'improving': 16,
             'them': 1788,
             'commented': 18,
             'on': 6741,
             'other': 1702,
             'topics': 10,
             'among': 370,
             'purchasing': 17,
             'departments': 25,
             'well': 897,
             'operated': 27,
             'follow': 97,
             'generally': 132,
             'accepted': 96,
             'practices': 53,
             'inure': 2,
             'best': 351,
             'both': 730,
             'governments': 61,
             'merger': 21,
             'proposed': 84,
             'however': 552,
             'believes': 43,
             'two': 1412,
             'offices': 45,
             'should': 888,
             'be': 6377,
             'combined': 40,
             'achieve': 51,
             'greater': 188,
             'efficiency': 50,
             'reduce': 62,
             'cost': 229,
             'administration': 161,
             'department': 225,
             'is': 10109,
             'lacking': 32,
             'experienced': 53,
             'clerical': 9,
             'personnel': 75,
             'as': 7253,
             'result': 244,
             'policies': 68,
             'urged': 35,
             'take': 610,
             'steps': 119,
             'remedy': 13,
             'problem': 313,
             'implementation': 8,
             'automobile': 50,
             'title': 77,
             'law': 299,
             'also': 1069,
             'outgoing': 8,
             'next': 394,
             'legislature': 39,
             'provide': 216,
             'enabling': 13,
             'funds': 95,
             're-set': 1,
             'effective': 129,
             'date': 103,
             'so': 1985,
             'orderly': 20,
             'may': 1402,
             'effected': 12,
             'swipe': 2,
             'at': 5372,
             'state': 807,
             'welfare': 53,
             "department's": 17,
             'handling': 38,
             'federal': 246,
             'granted': 56,
             'child': 213,
             'services': 139,
             'foster': 15,
             'homes': 62,
             'one': 3292,
             'major': 247,
             'items': 72,
             'general': 498,
             'assistance': 87,
             'program': 394,
             'but': 4381,
             'has': 2437,
             'seen': 279,
             'fit': 75,
             'distribute': 6,
             'through': 971,
             'all': 3001,
             'counties': 35,
             'with': 7289,
             'exception': 40,
             'receives': 20,
             'none': 108,
             'money': 265,
             'jurors': 4,
             'they': 3620,
             'realize': 69,
             'proportionate': 9,
             'distribution': 85,
             'might': 672,
             'disable': 1,
             'our': 1252,
             'less': 437,
             'populous': 5,
             'nevertheless': 73,
             'we': 2652,
             'feel': 216,
             'future': 227,
             'receive': 76,
             'some': 1618,
             'portion': 62,
             'available': 245,
             'failure': 89,
             'do': 1363,
             'will': 2245,
             'continue': 107,
             'disproportionate': 2,
             'burden': 44,
             'taxpayers': 21,
             "ordinary's": 1,
             'under': 707,
             'fire': 187,
             'its': 1858,
             'appointment': 28,
             'appraisers': 1,
             'guardians': 4,
             'administrators': 5,
             'awarding': 3,
             'fees': 29,
             'compensation': 17,
             'wards': 3,
             'protected': 31,
             'found': 536,
             'incorporated': 13,
             'into': 1791,
             'operating': 87,
             'procedures': 61,
             'recommendations': 22,
             'previous': 86,
             'juries': 1,
             'bar': 82,
             'association': 132,
             'interim': 11,
             'citizens': 86,
             'actions': 68,
             'serve': 107,
             'protect': 34,
             'fact': 447,
             'effect': 213,
             "court's": 8,
             'from': 4370,
             'undue': 13,
             'costs': 176,
             'appointed': 42,
             'elected': 33,
             'servants': 22,
             'unmeritorious': 1,
             'criticisms': 11,
             'regarding': 40,
             'new': 1635,
             'multi-million-dollar': 2,
             'airport': 19,
             'when': 2331,
             'management': 91,
             'takes': 86,
             'jan.': 20,
             '1': 527,
             'eliminate': 26,
             'political': 258,
             'influences': 14,
             'not': 4610,
             'elaborate': 32,
             'added': 172,
             'there': 2728,
             'periodic': 9,
             'surveillance': 6,
             'pricing': 7,
             'concessionaires': 7,
             'purpose': 149,
             'keeping': 60,
             'prices': 61,
             'reasonable': 64,
             'ask': 128,
             'jail': 21,
             'deputies': 13,
             'matters': 64,
             ':': 1795,
             '(': 2435,
             ')': 2466,
             'four': 360,
             'additional': 120,
             'employed': 49,
             'doctor': 100,
             'medical': 162,
             'intern': 2,
             'extern': 1,
             'night': 411,
             'weekend': 27,
             'duty': 61,
             '2': 446,
             'work': 762,
             'officials': 62,
             'pass': 89,
             'legislation': 46,
             'permit': 77,
             'establishment': 52,
             'fair': 78,
             'equitable': 11,
             'pension': 13,
             'plan': 205,
             'employes': 17,
             'praised': 13,
             'operation': 113,
             'police': 155,
             'tax': 201,
             "commissioner's": 1,
             'office': 255,
             'bellwood': 1,
             'alpharetta': 1,
             'prison': 42,
             'farms': 16,
             'grady': 5,
             'hospital': 110,
             'health': 105,
             'mayor': 38,
             'william': 148,
             'b.': 76,
             'hartsfield': 5,
             'filed': 33,
             'suit': 48,
             'divorce': 29,
             'his': 6996,
             'wife': 228,
             'pearl': 9,
             'williams': 32,
             'petition': 15,
             'mental': 43,
             'cruelty': 13,
             'couple': 122,
             'married': 105,
             'aug.': 25,
             '1913': 12,
             'son': 165,
             'berry': 9,
             'daughter': 72,
             'mrs.': 534,
             'j.': 120,
             'm.': 62,
             'cheshire': 1,
             'griffin': 4,
             'attorneys': 9,
             'amicable': 1,
             'property': 156,
             'settlement': 26,
             'agreed': 81,
             'upon': 495,
             'listed': 44,
             "mayor's": 9,
             'occupation': 24,
             'attorney': 65,
             'age': 227,
             '71': 10,
             "wife's": 15,
             '74': 6,
             'birth': 66,
             'opelika': 2,
             'ala.': 7,
             'lived': 115,
             'together': 267,
             'man': 1207,
             'more': 2215,
             'than': 1790,
             'year': 658,
             'home': 547,
             '637': 1,
             'e.': 86,
             'pelham': 5,
             'rd.': 3,
             'aj': 118,
             'henry': 83,
             'l.': 56,
             'bowden': 2,
             'brief': 73,
             'interlude': 5,
             'since': 628,
             '1937': 10,
             'career': 67,
             'goes': 89,
             'back': 966,
             'council': 103,
             '1923': 7,
             'present': 377,
             'expires': 1,
             'he': 9548,
             'succeeded': 33,
             'who': 2252,
             'became': 246,
             'candidate': 34,
             'sept.': 34,
             '13': 49,
             'after': 1069,
             'announced': 88,
             'would': 2714,
             'run': 212,
             'reelection': 2,
             'georgia': 46,
             'republicans': 29,
             'getting': 164,
             'strong': 202,
             'encouragement': 14,
             'enter': 78,
             '1962': 35,
             "governor's": 14,
             'race': 103,
             'top': 204,
             'official': 75,
             'wednesday': 35,
             'robert': 83,
             'snodgrass': 2,
             'gop': 13,
             'chairman': 67,
             'meeting': 159,
             'held': 264,
             'tuesday': 58,
             'blue': 143,
             'ridge': 18,
             'brought': 253,
             'enthusiastic': 24,
             'responses': 28,
             'audience': 115,
             'party': 216,
             'james': 101,
             'w.': 84,
             'dorsey': 1,
             'enthusiasm': 28,
             'picking': 14,
             'up': 1890,
             'rally': 10,
             '8': 106,
             'savannah': 9,
             'newly': 28,
             'texas': 69,
             'sen.': 30,
             'john': 362,
             'tower': 13,
             'featured': 8,
             'speaker': 49,
             'warned': 22,
             'entering': 24,
             'governor': 83,
             'force': 230,
             'petitions': 8,
             'out': 2097,
             'voting': 30,
             'precincts': 5,
             'obtain': 42,
             'signatures': 5,
             'registered': 23,
             'despite': 104,
             'warning': 44,
             'unanimous': 5,
             'vote': 75,
             'according': 140,
             'attended': 36,
             'crowd': 53,
             'asked': 398,
             'whether': 286,
             'wanted': 226,
             'wait': 94,
             'make': 794,
             'voted': 27,
             '--': 3432,
             'were': 3284,
             'dissents': 2,
             'largest': 53,
             'hurdle': 3,
             'face': 371,
             'says': 200,
             'before': 1016,
             'making': 255,
             'first': 1361,
             'alternative': 34,
             'courses': 61,
             'must': 1013,
             'taken': 281,
             'five': 286,
             'per': 371,
             'cent': 155,
             'each': 877,
             'sign': 94,
             'requesting': 8,
             'allowed': 86,
             'names': 89,
             'candidates': 38,
             'ballot': 12,
             'hold': 169,
             'unit': 103,
             'system': 416,
             'opposes': 2,
             'platform': 72,
             'sam': 79,
             'caldwell': 3,
             'highway': 40,
             'public': 438,
             'relations': 102,
             'director': 101,
             'resigned': 9,
             'lt.': 8,
             'gov.': 19,
             'garland': 9,
             "byrd's": 3,
             'campaign': 81,
             "caldwell's": 2,
             'resignation': 7,
             'expected': 187,
             'time': 1598,
             'rob': 19,
             'ledford': 1,
             'gainesville': 1,
             'assistant': 36,
             'three': 610,
             'years': 950,
             'gubernatorial': 7,
             'starts': 31,
             'become': 359,
             'coordinator': 5,
             'byrd': 9,
             'wind': 63,
             '1961': 134,
             'session': 80,
             'monday': 68,
             'head': 424,
             'where': 937,
             'bond': 46,
             'approved': 40,
             'shortly': 34,
             'adjournment': 4,
             'afternoon': 106,
             'senate': 62,
             'approve': 14,
             'study': 246,
             'allotted': 10,
             'rural': 54,
             'urban': 42,
             'areas': 236,
             'determine': 107,
             'what': 1908,
             'adjustments': 20,
             'made': 1125,
             'vandiver': 6,
             'traditional': 78,
             'visit': 109,
             'chambers': 11,
             'toward': 386,
             'likely': 151,
             'mention': 50,
             '$100': 12,
             'million': 204,
             'issue': 152,
             'earlier': 146,
             'priority': 18,
             'item': 55,
             'construction': 95,
             'bonds': 47,
             'meanwhile': 35,
             'learned': 117,
             'very': 796,
             'near': 198,
             'being': 712,
             'ready': 143,
             '$30': 2,
             'worth': 94,
             'reconstruction': 11,
             'go': 626,
             'courts': 50,
             'friendly': 61,
             'test': 119,
             'validity': 15,
             'then': 1380,
             'sales': 133,
             'begin': 84,
             'contracts': 24,
             'let': 384,
             'repair': 20,
             'most': 1159,
             'heavily': 60,
             'traveled': 22,
             'highways': 16,
             'source': 94,
             '$3': 3,
             '$4': 5,
             'roads': 58,
             'authority': 93,
             'road': 197,
             'revolving': 6,
             'fund': 62,
             'apparently': 125,
             'intends': 6,
             'issued': 50,
             'every': 491,
             'old': 661,
             'ones': 116,
             'paid': 145,
             'off': 639,
             'authorities': 39,
             'opened': 131,
             '1958': 91,
             'battle': 87,
             'against': 627,
             'issuance': 7,
             '$50': 7,
             'marvin': 9,
             'told': 413,
             'constitution': 49,
             'consulted': 17,
             'yet': 419,
             'about': 1815,
             'plans': 113,
             'schley': 1,
             'rep.': 13,
             'd.': 65,
             'offer': 80,
             'resolution': 64,
             'house': 591,
             'rescind': 2,
             "body's": 3,
             'action': 291,
             'itself': 304,
             '$10': 7,
             'day': 687,
             'increase': 195,
             'expense': 50,
             'allowances': 25,
             'sunday': 101,
             'research': 171,
             'done': 319,
             'quickie': 2,
             'can': 1772,
             'repealed': 3,
             'outright': 9,
             'notice': 59,
             'given': 377,
             'reconsideration': 4,
             'sought': 55,
             'while': 680,
             'emphasizing': 4,
             'technical': 120,
             'details': 57,
             'fully': 80,
             'worked': 128,
             'seek': 69,
             'set': 414,
             'aside': 67,
             'privilege': 18,
             '87-31': 1,
             'similar': 157,
             'passed': 157,
             '29-5': 1,
             'word': 274,
             'offered': 83,
             'pointed': 74,
             'last': 676,
             'november': 74,
             'rejected': 33,
             'constitutional': 25,
             'amendment': 23,
             'allow': 72,
             'pay': 172,
             'raises': 16,
             'sessions': 26,
             'veteran': 27,
             'jackson': 36,
             'legislator': 7,
             'aid': 134,
             'education': 214,
             'something': 450,
             'consistently': 19,
             'opposed': 41,
             'past': 281,
             'mac': 1,
             'barber': 8,
             'commerce': 58,
             'asking': 67,
             'endorse': 6,
             'increased': 146,
             'support': 180,
             'provided': 132,
             'expended': 12,
             '13th': 6,
             'members': 325,
             'congressional': 22,
             'delegation': 11,
             'washington': 206,
             'like': 1292,
             'see': 772,
             'congressmen': 10,
             'specifically': 38,
             'him': 2619,
             'tossed': 31,
             'hopper': 2,
             'formally': 18,
             'read': 174,
             'event': 81,
             'congress': 152,
             'does': 485,
             'board': 239,
             'directed': 68,
             'give': 389,
             'teacher': 80,
             'colquitt': 2,
             'long': 752,
             'hot': 130,
             'controversy': 26,
             'miller': 22,
             'school': 493,
             'superintendent': 17,
             'policeman': 19,
             'put': 437,
             'coolest': 4,
             'i': 5164,
             'ever': 344,
             'saw': 352,
             'harry': 35,
             'davis': 27,
             'agriculture': 23,
             'defeated': 15,
             'felix': 30,
             'bush': 14,
             'principal': 92,
             'democratic': 109,
             '1,119': 1,
             'votes': 20,
             "saturday's": 3,
             'got': 482,
             '402': 1,
             'ordinary': 72,
             'carey': 5,
             'armed': 60,
             'pistol': 27,
             'stood': 212,
             'polls': 10,
             'insure': 24,
             'order': 376,
             'calmest': 1,
             'tom': 63,
             'just': 872,
             'church': 348,
             "didn't": 401,
             'smell': 34,
             'drop': 59,
             'liquor': 43,
             'bit': 101,
             'trouble': 134,
             'leading': 68,
             'quiet': 76,
             'marked': 85,
             'anonymous': 17,
             'midnight': 23,
             'phone': 54,
             'calls': 70,
             'veiled': 6,
             'threats': 14,
             'violence': 46,
             'former': 131,
             'george': 129,
             'p.': 58,
             'callan': 1,
             'shot': 113,
             'himself': 603,
             'death': 277,
             'march': 121,
             '18': 55,
             'days': 384,
             'post': 85,
             'dispute': 34,
             'during': 585,
             'reportedly': 9,
             'telephone': 76,
             'too': 834,
             'subjected': 24,
             'soon': 199,
             'scheduled': 38,
             'local': 288,
             'feared': 14,
             'carry': 88,
             'gun': 118,
             'promised': 45,
             'sheriff': 20,
             'tabb': 1,
             'good': 806,
             'promise': 45,
             'everything': 185,
             'went': 507,
             'real': 258,
             'smooth': 42,
             "wasn't": 154,
             'austin': 18,
             'approval': 51,
             'price': 108,
             "daniel's": 1,
             'abandoned': 25,
             'seemed': 333,
             'certain': 313,
             'thursday': 33,
             'adamant': 5,
             'protests': 11,
             'bankers': 15,
             'daniel': 14,
             'personally': 40,
             'led': 132,
             'fight': 98,
             'measure': 91,
             'watered': 7,
             'down': 895,
             'considerably': 44,
             'rejection': 11,
             'legislatures': 2,
             'hearing': 76,
             'revenue': 35,
             'taxation': 11,
             'rules': 85,
             'automatically': 36,
             'subcommittee': 5,
             'week': 275,
             'questions': 140,
             'taunted': 2,
             'appearing': 16,
             'witnesses': 21,
             'left': 480,
             'little': 831,
             'doubt': 114,
             'recommend': 25,
             'passage': 49,
             'termed': 15,
             'extremely': 50,
             'conservative': 31,
             'estimate': 39,
             'produce': 82,
             '17': 41,
             'dollars': 97,
             'help': 311,
             'erase': 1,
             'anticipated': 23,
             'deficit': 12,
             '63': 5,
             'current': 104,
             'fiscal': 120,
             '31': 38,
             'merely': 135,
             'means': 310,
             'enforcing': 5,
             'escheat': 2,
             'books': 96,
             'republic': 43,
             'permits': 27,
             'over': 1236,
             'bank': 83,
             'accounts': 38,
             'stocks': 18,
             'personal': 196,
             'persons': 121,
             'missing': 33,
             'seven': 113,
             'bill': 143,
             'drafted': 5,
             'banks': 37,
             'insurance': 46,
             'firms': 55,
             'pipeline': 6,
             'companies': 87,
             'corporations': 25,
             'report': 174,
             'treasurer': 14,
             'cannot': 258,
             'enforced': 20,
             'now': 1314,
             'because': 883,
             'almost': 432,
             'impossible': 84,
             'locate': 16,
             'declared': 66,
             'dewey': 3,
             'lawrence': 39,
             'tyler': 2,
             'lawyer': 43,
             'representing': 30,
             'sounded': 35,
             'opposition': 46,
             'keynote': 4,
             'violate': 7,
             'their': 2669,
             'contractual': 7,
             'obligations': 22,
             'depositors': 1,
             'undermine': 8,
             'confidence': 56,
             'customers': 40,
             'if': 2198,
             'you': 3286,
             'destroy': 48,
             'economy': 79,
             'circulation': 16,
             'millions': 49,
             'charles': 96,
             'hughes': 27,
             'sherman': 29,
             'sponsor': 22,
             'enact': 7,
             'amount': 172,
             'gift': 33,
             "taxpayers'": 2,
             'pockets': 17,
             'contention': 9,
             'denied': 47,
             'several': 377,
             'including': 171,
             'scott': 16,
             'hudson': 53,
             'gaynor': 1,
             'jones': 72,
             'houston': 25,
             'brady': 1,
             'harlingen': 1,
             'howard': 32,
             'cox': 5,
             'argued': 29,
             'probably': 261,
             'unconstitutional': 2,
             'impair': 4,
             'complained': 22,
             'enough': 430,
             'introduced': 52,
             'senators': 10,
             'unanimously': 11,
             'parkhouse': 5,
             'dallas': 58,
             'authorizing': 5,
             'schools': 195,
             'deaf': 12,
             'designed': 108,
             'special': 250,
             'schooling': 5,
             'students': 213,
             'scholastic': 9,
             'reduced': 79,
             'debate': 32,
             'authorize': 5,
             'agency': 56,
             'establish': 58,
             'county-wide': 2,
             '300,000': 6,
             'population': 136,
             'require': 86,
             'children': 355,
             'between': 730,
             '6': 114,
             'attend': 54,
             'permitting': 9,
             'older': 93,
             'residential': 45,
             'here': 750,
             'budget': 59,
             'harris': 28,
             'bexar': 1,
             'tarrant': 1,
             'el': 20,
             'paso': 11,
             '$451,500': 1,
             'savings': 23,
             '$157,460': 1,
             'yearly': 12,
             "year's": 43,
             'capital': 85,
             'outlay': 2,
             '$88,000': 1,
             'absorbed': 24,
             'tea': 29,
             'estimated': 67,
             '182': 1,
             'scholastics': 1,
             'saving': 21,
             'coming': 174,
             'live': 177,
             'get': 749,
             'hear': 153,
             'horse': 117,
             'parimutuels': 1,
             'reps.': 2,
             'v.': 31,
             'red': 197,
             'joe': 55,
             'ratcliff': 3,
             'still': 782,
             'expects': 22,
             'tell': 268,
             'folks': 18,
             ...})




```python
from collections import defaultdict
word_to_id = defaultdict(lambda:1,{word:i for i,word in enumerate(all_words)})
tag_to_id = {tag:i for i,tag in enumerate(all_tags)}
```


```python
word_to_id
```




    defaultdict(<function __main__.<lambda>>,
                {'#EOS#': 0,
                 '#UNK#': 1,
                 'the': 2,
                 ',': 3,
                 '.': 4,
                 'of': 5,
                 'and': 6,
                 'to': 7,
                 'a': 8,
                 'in': 9,
                 'that': 10,
                 'is': 11,
                 'was': 12,
                 'he': 13,
                 'for': 14,
                 '``': 15,
                 "''": 16,
                 'it': 17,
                 'with': 18,
                 'as': 19,
                 'his': 20,
                 'on': 21,
                 'be': 22,
                 ';': 23,
                 'at': 24,
                 'by': 25,
                 'i': 26,
                 'this': 27,
                 'had': 28,
                 '?': 29,
                 'not': 30,
                 'are': 31,
                 'but': 32,
                 'from': 33,
                 'or': 34,
                 'have': 35,
                 'an': 36,
                 'they': 37,
                 'which': 38,
                 '--': 39,
                 'one': 40,
                 'you': 41,
                 'were': 42,
                 'her': 43,
                 'all': 44,
                 'she': 45,
                 'there': 46,
                 'would': 47,
                 'their': 48,
                 'we': 49,
                 'him': 50,
                 'been': 51,
                 ')': 52,
                 'has': 53,
                 '(': 54,
                 'when': 55,
                 'who': 56,
                 'will': 57,
                 'more': 58,
                 'if': 59,
                 'no': 60,
                 'out': 61,
                 'so': 62,
                 'said': 63,
                 'what': 64,
                 'up': 65,
                 'its': 66,
                 'about': 67,
                 ':': 68,
                 'into': 69,
                 'than': 70,
                 'them': 71,
                 'can': 72,
                 'only': 73,
                 'other': 74,
                 'new': 75,
                 'some': 76,
                 'could': 77,
                 'time': 78,
                 '!': 79,
                 'these': 80,
                 'two': 81,
                 'may': 82,
                 'then': 83,
                 'do': 84,
                 'first': 85,
                 'any': 86,
                 'my': 87,
                 'now': 88,
                 'such': 89,
                 'like': 90,
                 'our': 91,
                 'over': 92,
                 'man': 93,
                 'me': 94,
                 'even': 95,
                 'most': 96,
                 'made': 97,
                 'also': 98,
                 'after': 99,
                 'did': 100,
                 'many': 101,
                 'before': 102,
                 'must': 103,
                 'af': 104,
                 'through': 105,
                 'back': 106,
                 'years': 107,
                 'where': 108,
                 'much': 109,
                 'your': 110,
                 'way': 111,
                 'well': 112,
                 'down': 113,
                 'should': 114,
                 'because': 115,
                 'each': 116,
                 'just': 117,
                 'those': 118,
                 'people': 119,
                 'mr.': 120,
                 'too': 121,
                 'how': 122,
                 'little': 123,
                 'state': 124,
                 'good': 125,
                 'very': 126,
                 'make': 127,
                 'world': 128,
                 'still': 129,
                 'see': 130,
                 'own': 131,
                 'men': 132,
                 'work': 133,
                 'long': 134,
                 'here': 135,
                 'get': 136,
                 'both': 137,
                 'between': 138,
                 'life': 139,
                 'being': 140,
                 'under': 141,
                 'never': 142,
                 'day': 143,
                 'same': 144,
                 'another': 145,
                 'know': 146,
                 'while': 147,
                 'last': 148,
                 'us': 149,
                 'might': 150,
                 'great': 151,
                 'old': 152,
                 'year': 153,
                 'off': 154,
                 'come': 155,
                 'since': 156,
                 'against': 157,
                 'go': 158,
                 'came': 159,
                 'right': 160,
                 'used': 161,
                 'take': 162,
                 'three': 163,
                 'himself': 164,
                 'states': 165,
                 'few': 166,
                 'house': 167,
                 'use': 168,
                 'during': 169,
                 'without': 170,
                 'again': 171,
                 'place': 172,
                 'american': 173,
                 'around': 174,
                 'however': 175,
                 'home': 176,
                 'small': 177,
                 'found': 178,
                 'mrs.': 179,
                 '1': 180,
                 'thought': 181,
                 'went': 182,
                 'say': 183,
                 'part': 184,
                 'once': 185,
                 'general': 186,
                 'high': 187,
                 'upon': 188,
                 'school': 189,
                 'every': 190,
                 "don't": 191,
                 'does': 192,
                 'got': 193,
                 'united': 194,
                 'left': 195,
                 'number': 196,
                 'course': 197,
                 'war': 198,
                 'until': 199,
                 'always': 200,
                 'away': 201,
                 'something': 202,
                 'fact': 203,
                 '2': 204,
                 'water': 205,
                 'though': 206,
                 'public': 207,
                 'less': 208,
                 'put': 209,
                 'think': 210,
                 'almost': 211,
                 'hand': 212,
                 'enough': 213,
                 'took': 214,
                 'far': 215,
                 'head': 216,
                 'yet': 217,
                 'government': 218,
                 'system': 219,
                 'set': 220,
                 'better': 221,
                 'told': 222,
                 'night': 223,
                 'nothing': 224,
                 'end': 225,
                 'why': 226,
                 "didn't": 227,
                 'called': 228,
                 'eyes': 229,
                 'find': 230,
                 'going': 231,
                 'look': 232,
                 'asked': 233,
                 'later': 234,
                 'knew': 235,
                 'point': 236,
                 'next': 237,
                 'program': 238,
                 'city': 239,
                 'business': 240,
                 'group': 241,
                 'give': 242,
                 'toward': 243,
                 'young': 244,
                 'let': 245,
                 'days': 246,
                 'room': 247,
                 'president': 248,
                 'side': 249,
                 'social': 250,
                 'present': 251,
                 'given': 252,
                 'several': 253,
                 'order': 254,
                 'national': 255,
                 'possible': 256,
                 'rather': 257,
                 'second': 258,
                 'face': 259,
                 'per': 260,
                 'among': 261,
                 'form': 262,
                 'often': 263,
                 'important': 264,
                 'things': 265,
                 'looked': 266,
                 'early': 267,
                 'white': 268,
                 'john': 269,
                 'case': 270,
                 'large': 271,
                 'four': 272,
                 'need': 273,
                 'big': 274,
                 'become': 275,
                 'within': 276,
                 'felt': 277,
                 'children': 278,
                 'along': 279,
                 'saw': 280,
                 'best': 281,
                 'church': 282,
                 'ever': 283,
                 'least': 284,
                 'power': 285,
                 'development': 286,
                 'seemed': 287,
                 'thing': 288,
                 'light': 289,
                 'family': 290,
                 'interest': 291,
                 'want': 292,
                 'members': 293,
                 'mind': 294,
                 'area': 295,
                 'country': 296,
                 'others': 297,
                 'although': 298,
                 'turned': 299,
                 'done': 300,
                 'open': 301,
                 "'": 302,
                 'god': 303,
                 'service': 304,
                 'problem': 305,
                 'certain': 306,
                 'kind': 307,
                 'different': 308,
                 'thus': 309,
                 'began': 310,
                 'door': 311,
                 'help': 312,
                 'sense': 313,
                 'means': 314,
                 'whole': 315,
                 'matter': 316,
                 'perhaps': 317,
                 'itself': 318,
                 'york': 319,
                 "it's": 320,
                 'times': 321,
                 'law': 322,
                 'human': 323,
                 'line': 324,
                 'above': 325,
                 'name': 326,
                 'example': 327,
                 'action': 328,
                 'company': 329,
                 'hands': 330,
                 'local': 331,
                 'show': 332,
                 '3': 333,
                 'whether': 334,
                 'five': 335,
                 'history': 336,
                 'gave': 337,
                 'today': 338,
                 'either': 339,
                 'act': 340,
                 'feet': 341,
                 'across': 342,
                 'taken': 343,
                 'past': 344,
                 'quite': 345,
                 'anything': 346,
                 'seen': 347,
                 'having': 348,
                 'death': 349,
                 'experience': 350,
                 'body': 351,
                 'week': 352,
                 'half': 353,
                 'really': 354,
                 'word': 355,
                 'field': 356,
                 'car': 357,
                 'words': 358,
                 'already': 359,
                 'themselves': 360,
                 "i'm": 361,
                 'information': 362,
                 'tell': 363,
                 'shall': 364,
                 'together': 365,
                 'college': 366,
                 'money': 367,
                 'period': 368,
                 'held': 369,
                 'keep': 370,
                 'sure': 371,
                 'probably': 372,
                 'free': 373,
                 'seems': 374,
                 'political': 375,
                 'real': 376,
                 'cannot': 377,
                 'behind': 378,
                 'question': 379,
                 'air': 380,
                 'office': 381,
                 'making': 382,
                 'brought': 383,
                 'miss': 384,
                 'whose': 385,
                 'special': 386,
                 'major': 387,
                 'heard': 388,
                 'problems': 389,
                 'federal': 390,
                 'became': 391,
                 'study': 392,
                 'ago': 393,
                 'moment': 394,
                 'available': 395,
                 'known': 396,
                 'result': 397,
                 'street': 398,
                 'economic': 399,
                 'boy': 400,
                 'position': 401,
                 'reason': 402,
                 'change': 403,
                 'south': 404,
                 'board': 405,
                 'individual': 406,
                 'job': 407,
                 'am': 408,
                 'society': 409,
                 'areas': 410,
                 'west': 411,
                 'close': 412,
                 'turn': 413,
                 'community': 414,
                 'true': 415,
                 'love': 416,
                 'court': 417,
                 'force': 418,
                 'full': 419,
                 'cost': 420,
                 'seem': 421,
                 'wife': 422,
                 'future': 423,
                 'age': 424,
                 'wanted': 425,
                 'voice': 426,
                 'department': 427,
                 'center': 428,
                 'woman': 429,
                 'control': 430,
                 'common': 431,
                 'policy': 432,
                 'necessary': 433,
                 'following': 434,
                 'front': 435,
                 'sometimes': 436,
                 'six': 437,
                 'girl': 438,
                 'clear': 439,
                 'further': 440,
                 'land': 441,
                 'provide': 442,
                 'feel': 443,
                 'party': 444,
                 'able': 445,
                 'mother': 446,
                 'music': 447,
                 'education': 448,
                 'university': 449,
                 'child': 450,
                 'effect': 451,
                 'students': 452,
                 'level': 453,
                 'run': 454,
                 'stood': 455,
                 'military': 456,
                 'town': 457,
                 'short': 458,
                 'morning': 459,
                 'total': 460,
                 'outside': 461,
                 'rate': 462,
                 'figure': 463,
                 'art': 464,
                 'century': 465,
                 'class': 466,
                 'washington': 467,
                 '4': 468,
                 'north': 469,
                 'usually': 470,
                 'plan': 471,
                 'leave': 472,
                 'therefore': 473,
                 'evidence': 474,
                 'top': 475,
                 'million': 476,
                 'sound': 477,
                 'black': 478,
                 'strong': 479,
                 'hard': 480,
                 'tax': 481,
                 'various': 482,
                 'says': 483,
                 'believe': 484,
                 'type': 485,
                 'value': 486,
                 'play': 487,
                 'surface': 488,
                 'soon': 489,
                 'mean': 490,
                 'near': 491,
                 'lines': 492,
                 'table': 493,
                 'peace': 494,
                 'modern': 495,
                 'road': 496,
                 'red': 497,
                 'book': 498,
                 'personal': 499,
                 'process': 500,
                 'situation': 501,
                 'minutes': 502,
                 'increase': 503,
                 'schools': 504,
                 'idea': 505,
                 'english': 506,
                 'alone': 507,
                 'women': 508,
                 'gone': 509,
                 'nor': 510,
                 'living': 511,
                 'america': 512,
                 'started': 513,
                 'longer': 514,
                 'dr.': 515,
                 'cut': 516,
                 'finally': 517,
                 'secretary': 518,
                 'nature': 519,
                 'private': 520,
                 'third': 521,
                 'months': 522,
                 'section': 523,
                 'greater': 524,
                 'call': 525,
                 'fire': 526,
                 'expected': 527,
                 'needed': 528,
                 "that's": 529,
                 'kept': 530,
                 'ground': 531,
                 'view': 532,
                 'values': 533,
                 'everything': 534,
                 'pressure': 535,
                 'dark': 536,
                 'basis': 537,
                 'space': 538,
                 'east': 539,
                 'father': 540,
                 'required': 541,
                 'union': 542,
                 'spirit': 543,
                 'complete': 544,
                 'except': 545,
                 'wrote': 546,
                 "i'll": 547,
                 'moved': 548,
                 'support': 549,
                 'return': 550,
                 'conditions': 551,
                 'recent': 552,
                 'attention': 553,
                 'late': 554,
                 'particular': 555,
                 'live': 556,
                 'hope': 557,
                 'costs': 558,
                 'else': 559,
                 'brown': 560,
                 'taking': 561,
                 "couldn't": 562,
                 'forces': 563,
                 'nations': 564,
                 'beyond': 565,
                 'stage': 566,
                 'read': 567,
                 'report': 568,
                 'coming': 569,
                 'hours': 570,
                 'person': 571,
                 'inside': 572,
                 'dead': 573,
                 'material': 574,
                 'instead': 575,
                 'lost': 576,
                 'heart': 577,
                 'looking': 578,
                 'low': 579,
                 'miles': 580,
                 'data': 581,
                 'added': 582,
                 'pay': 583,
                 'amount': 584,
                 'followed': 585,
                 'feeling': 586,
                 '1960': 587,
                 'single': 588,
                 'makes': 589,
                 'research': 590,
                 'including': 591,
                 'basic': 592,
                 'hundred': 593,
                 'move': 594,
                 'industry': 595,
                 'cold': 596,
                 'simply': 597,
                 'developed': 598,
                 'tried': 599,
                 'hold': 600,
                 "can't": 601,
                 'reached': 602,
                 'committee': 603,
                 'island': 604,
                 'defense': 605,
                 'equipment': 606,
                 'actually': 607,
                 'shown': 608,
                 'son': 609,
                 'central': 610,
                 'religious': 611,
                 'river': 612,
                 'getting': 613,
                 'st.': 614,
                 'beginning': 615,
                 'sort': 616,
                 'ten': 617,
                 'received': 618,
                 '&': 619,
                 'doing': 620,
                 'terms': 621,
                 'trying': 622,
                 'rest': 623,
                 'medical': 624,
                 'u.s.': 625,
                 'care': 626,
                 'especially': 627,
                 'friends': 628,
                 'picture': 629,
                 'indeed': 630,
                 'administration': 631,
                 'fine': 632,
                 'subject': 633,
                 'difficult': 634,
                 'building': 635,
                 'higher': 636,
                 'wall': 637,
                 'simple': 638,
                 'meeting': 639,
                 'walked': 640,
                 'floor': 641,
                 'foreign': 642,
                 'bring': 643,
                 'similar': 644,
                 'passed': 645,
                 'range': 646,
                 'paper': 647,
                 'property': 648,
                 'natural': 649,
                 'final': 650,
                 'training': 651,
                 'county': 652,
                 'police': 653,
                 'cent': 654,
                 'international': 655,
                 'growth': 656,
                 'market': 657,
                 "wasn't": 658,
                 'talk': 659,
                 'start': 660,
                 'england': 661,
                 'written': 662,
                 'hear': 663,
                 'suddenly': 664,
                 'story': 665,
                 'issue': 666,
                 'congress': 667,
                 'needs': 668,
                 '10': 669,
                 'answer': 670,
                 'hall': 671,
                 'likely': 672,
                 'working': 673,
                 'countries': 674,
                 'considered': 675,
                 "you're": 676,
                 'earth': 677,
                 'sat': 678,
                 'purpose': 679,
                 'meet': 680,
                 'labor': 681,
                 'results': 682,
                 'entire': 683,
                 'happened': 684,
                 'william': 685,
                 'cases': 686,
                 'stand': 687,
                 'difference': 688,
                 'production': 689,
                 'hair': 690,
                 'involved': 691,
                 'fall': 692,
                 'stock': 693,
                 'food': 694,
                 'earlier': 695,
                 'increased': 696,
                 'whom': 697,
                 'particularly': 698,
                 'paid': 699,
                 'sent': 700,
                 'effort': 701,
                 'knowledge': 702,
                 'hour': 703,
                 'letter': 704,
                 'club': 705,
                 'using': 706,
                 'below': 707,
                 'thinking': 708,
                 'yes': 709,
                 'christian': 710,
                 'blue': 711,
                 'ready': 712,
                 'bill': 713,
                 'deal': 714,
                 'points': 715,
                 'trade': 716,
                 'certainly': 717,
                 'ideas': 718,
                 'industrial': 719,
                 'square': 720,
                 'boys': 721,
                 'methods': 722,
                 'addition': 723,
                 'method': 724,
                 'bad': 725,
                 'due': 726,
                 '5': 727,
                 'girls': 728,
                 'moral': 729,
                 'decided': 730,
                 'reading': 731,
                 'statement': 732,
                 'weeks': 733,
                 'neither': 734,
                 'nearly': 735,
                 'directly': 736,
                 'showed': 737,
                 'throughout': 738,
                 'according': 739,
                 'questions': 740,
                 'color': 741,
                 'kennedy': 742,
                 'anyone': 743,
                 'try': 744,
                 'services': 745,
                 'programs': 746,
                 'nation': 747,
                 'lay': 748,
                 'french': 749,
                 'size': 750,
                 'remember': 751,
                 'physical': 752,
                 'record': 753,
                 'member': 754,
                 'comes': 755,
                 'understand': 756,
                 'southern': 757,
                 'western': 758,
                 'strength': 759,
                 'population': 760,
                 'normal': 761,
                 'merely': 762,
                 'district': 763,
                 'volume': 764,
                 'concerned': 765,
                 'appeared': 766,
                 'temperature': 767,
                 '1961': 768,
                 'aid': 769,
                 'trouble': 770,
                 'trial': 771,
                 'summer': 772,
                 'direction': 773,
                 'ran': 774,
                 'sales': 775,
                 'list': 776,
                 'continued': 777,
                 'friend': 778,
                 'evening': 779,
                 'maybe': 780,
                 'literature': 781,
                 'generally': 782,
                 'association': 783,
                 'provided': 784,
                 'led': 785,
                 'army': 786,
                 'met': 787,
                 'influence': 788,
                 'opened': 789,
                 'former': 790,
                 'science': 791,
                 'student': 792,
                 'step': 793,
                 'changes': 794,
                 'chance': 795,
                 'husband': 796,
                 'hot': 797,
                 'series': 798,
                 'average': 799,
                 'works': 800,
                 'month': 801,
                 'cause': 802,
                 'effective': 803,
                 'george': 804,
                 'planning': 805,
                 'systems': 806,
                 "wouldn't": 807,
                 'direct': 808,
                 'soviet': 809,
                 'stopped': 810,
                 'wrong': 811,
                 'lead': 812,
                 'myself': 813,
                 'piece': 814,
                 'theory': 815,
                 'ask': 816,
                 'worked': 817,
                 'freedom': 818,
                 'organization': 819,
                 'clearly': 820,
                 'movement': 821,
                 'ways': 822,
                 'press': 823,
                 'somewhat': 824,
                 'spring': 825,
                 'efforts': 826,
                 'consider': 827,
                 'meaning': 828,
                 'bed': 829,
                 'fear': 830,
                 'lot': 831,
                 'treatment': 832,
                 'beautiful': 833,
                 'note': 834,
                 'forms': 835,
                 'placed': 836,
                 'hotel': 837,
                 'truth': 838,
                 'apparently': 839,
                 'degree': 840,
                 'groups': 841,
                 "he's": 842,
                 'plant': 843,
                 'carried': 844,
                 'wide': 845,
                 "i've": 846,
                 'respect': 847,
                 "man's": 848,
                 'herself': 849,
                 'numbers': 850,
                 'manner': 851,
                 'reaction': 852,
                 'easy': 853,
                 'farm': 854,
                 'immediately': 855,
                 'running': 856,
                 'approach': 857,
                 'game': 858,
                 'recently': 859,
                 'larger': 860,
                 'lower': 861,
                 'charge': 862,
                 'couple': 863,
                 'de': 864,
                 'daily': 865,
                 'eye': 866,
                 'performance': 867,
                 'feed': 868,
                 'oh': 869,
                 'march': 870,
                 'persons': 871,
                 'understanding': 872,
                 'arms': 873,
                 'opportunity': 874,
                 'c': 875,
                 'blood': 876,
                 'additional': 877,
                 'j.': 878,
                 'technical': 879,
                 'fiscal': 880,
                 'radio': 881,
                 'described': 882,
                 'stop': 883,
                 'progress': 884,
                 'steps': 885,
                 'test': 886,
                 'chief': 887,
                 'reported': 888,
                 'served': 889,
                 'based': 890,
                 'main': 891,
                 'determined': 892,
                 'image': 893,
                 'decision': 894,
                 'window': 895,
                 'religion': 896,
                 'aj': 897,
                 'gun': 898,
                 'responsibility': 899,
                 'middle': 900,
                 'europe': 901,
                 'british': 902,
                 'character': 903,
                 'learned': 904,
                 'horse': 905,
                 'writing': 906,
                 'appear': 907,
                 's.': 908,
                 'account': 909,
                 'ones': 910,
                 'serious': 911,
                 'activity': 912,
                 'types': 913,
                 'green': 914,
                 'length': 915,
                 'lived': 916,
                 'audience': 917,
                 'letters': 918,
                 'returned': 919,
                 'obtained': 920,
                 'nuclear': 921,
                 'specific': 922,
                 'corner': 923,
                 'forward': 924,
                 'activities': 925,
                 'slowly': 926,
                 'doubt': 927,
                 '6': 928,
                 'justice': 929,
                 'moving': 930,
                 'latter': 931,
                 'gives': 932,
                 'straight': 933,
                 'hit': 934,
                 'plane': 935,
                 'quality': 936,
                 'design': 937,
                 'obviously': 938,
                 'operation': 939,
                 'plans': 940,
                 'shot': 941,
                 'seven': 942,
                 'a.': 943,
                 'choice': 944,
                 'poor': 945,
                 'staff': 946,
                 'function': 947,
                 'figures': 948,
                 'parts': 949,
                 'stay': 950,
                 'saying': 951,
                 'include': 952,
                 '15': 953,
                 'born': 954,
                 'pattern': 955,
                 '30': 956,
                 'cars': 957,
                 'whatever': 958,
                 'sun': 959,
                 'faith': 960,
                 'pool': 961,
                 'hospital': 962,
                 'corps': 963,
                 'wish': 964,
                 'lack': 965,
                 'completely': 966,
                 'heavy': 967,
                 'waiting': 968,
                 'speak': 969,
                 'ball': 970,
                 'standard': 971,
                 'extent': 972,
                 'visit': 973,
                 'democratic': 974,
                 'firm': 975,
                 'income': 976,
                 'ahead': 977,
                 'deep': 978,
                 "there's": 979,
                 'language': 980,
                 'principle': 981,
                 'none': 982,
                 'price': 983,
                 'designed': 984,
                 'indicated': 985,
                 'analysis': 986,
                 'distance': 987,
                 'expect': 988,
                 'established': 989,
                 'products': 990,
                 'effects': 991,
                 'growing': 992,
                 'importance': 993,
                 'continue': 994,
                 'serve': 995,
                 'determine': 996,
                 'cities': 997,
                 'elements': 998,
                 'negro': 999,
                 ...})




```python
tag_to_id
```




    {'#EOS#': 0,
     '#UNK#': 1,
     '.': 7,
     'ADJ': 13,
     'ADP': 4,
     'ADV': 2,
     'CONJ': 12,
     'DET': 6,
     'NOUN': 3,
     'NUM': 11,
     'PRON': 5,
     'PRT': 8,
     'VERB': 9,
     'X': 10}



convert words and tags into fixed-size matrix


```python
def to_matrix(lines,token_to_id,max_len=None,pad=0,dtype='int32',time_major=False):
    """Converts a list of names into rnn-digestable matrix with paddings added after the end"""
    
    max_len = max_len or max(map(len,lines))
    matrix = np.empty([len(lines),max_len],dtype)
    matrix.fill(pad)

    for i in range(len(lines)):
        line_ix = list(map(token_to_id.__getitem__,lines[i]))[:max_len]
        matrix[i,:len(line_ix)] = line_ix

    return matrix.T if time_major else matrix
```


```python
batch_words,batch_tags = zip(*[zip(*sentence) for sentence in data[-3:]])
print(batch_words, batch_tags)
print("Word ids:")
print(to_matrix(batch_words,word_to_id))
print("Tag ids:")
print(to_matrix(batch_tags,tag_to_id))
```

    (('the', 'doors', 'of', 'the', 'd', 'train', 'slid', 'shut', ',', 'and', 'as', 'i', 'dropped', 'into', 'a', 'seat', 'and', ',', 'exhaling', ',', 'looked', 'up', 'across', 'the', 'aisle', ',', 'the', 'whole', 'aviary', 'in', 'my', 'head', 'burst', 'into', 'song', '.'), ('she', 'was', 'a', 'living', 'doll', 'and', 'no', 'mistake', '--', 'the', 'blue-black', 'bang', ',', 'the', 'wide', 'cheekbones', ',', 'olive-flushed', ',', 'that', 'betrayed', 'the', 'cherokee', 'strain', 'in', 'her', 'midwestern', 'lineage', ',', 'and', 'the', 'mouth', 'whose', 'only', 'fault', ',', 'in', 'the', "novelist's", 'carping', 'phrase', ',', 'was', 'that', 'the', 'lower', 'lip', 'was', 'a', 'trifle', 'too', 'voluptuous', '.'), ('from', 'what', 'i', 'was', 'able', 'to', 'gauge', 'in', 'a', 'swift', ',', 'greedy', 'glance', ',', 'the', 'figure', 'inside', 'the', 'coral-colored', 'boucle', 'dress', 'was', 'stupefying', '.')) (('DET', 'NOUN', 'ADP', 'DET', 'NOUN', 'NOUN', 'VERB', 'VERB', '.', 'CONJ', 'ADP', 'PRON', 'VERB', 'ADP', 'DET', 'NOUN', 'CONJ', '.', 'VERB', '.', 'VERB', 'PRT', 'ADP', 'DET', 'NOUN', '.', 'DET', 'ADJ', 'NOUN', 'ADP', 'DET', 'NOUN', 'VERB', 'ADP', 'NOUN', '.'), ('PRON', 'VERB', 'DET', 'VERB', 'NOUN', 'CONJ', 'DET', 'NOUN', '.', 'DET', 'ADJ', 'NOUN', '.', 'DET', 'ADJ', 'NOUN', '.', 'ADJ', '.', 'PRON', 'VERB', 'DET', 'NOUN', 'NOUN', 'ADP', 'DET', 'ADJ', 'NOUN', '.', 'CONJ', 'DET', 'NOUN', 'DET', 'ADJ', 'NOUN', '.', 'ADP', 'DET', 'NOUN', 'VERB', 'NOUN', '.', 'VERB', 'ADP', 'DET', 'ADJ', 'NOUN', 'VERB', 'DET', 'NOUN', 'ADV', 'ADJ', '.'), ('ADP', 'DET', 'PRON', 'VERB', 'ADJ', 'ADP', 'NOUN', 'ADP', 'DET', 'ADJ', '.', 'ADJ', 'NOUN', '.', 'DET', 'NOUN', 'ADP', 'DET', 'ADJ', 'NOUN', 'NOUN', 'VERB', 'VERB', '.'))
    Word ids:
    [[   2 3057    5    2 2238 1334 4238 2454    3    6   19   26 1070   69
         8 2088    6    3    1    3  266   65  342    2    1    3    2  315
         1    9   87  216 3322   69 1558    4    0    0    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0]
     [  45   12    8  511 8419    6   60 3246   39    2    1    1    3    2
       845    1    3    1    3   10 9910    2    1 3470    9   43    1    1
         3    6    2 1046  385   73 4562    3    9    2    1    1 3250    3
        12   10    2  861 5240   12    8 8936  121    1    4]
     [  33   64   26   12  445    7 7346    9    8 3337    3    1 2811    3
         2  463  572    2    1    1 1649   12    1    4    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0]]
    Tag ids:
    [[ 6  3  4  6  3  3  9  9  7 12  4  5  9  4  6  3 12  7  9  7  9  8  4  6
       3  7  6 13  3  4  6  3  9  4  3  7  0  0  0  0  0  0  0  0  0  0  0  0
       0  0  0  0  0]
     [ 5  9  6  9  3 12  6  3  7  6 13  3  7  6 13  3  7 13  7  5  9  6  3  3
       4  6 13  3  7 12  6  3  6 13  3  7  4  6  3  9  3  7  9  4  6 13  3  9
       6  3  2 13  7]
     [ 4  6  5  9 13  4  3  4  6 13  7 13  3  7  6  3  4  6 13  3  3  9  9  7
       0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
       0  0  0  0  0]]


### Build model

Unlike our previous lab, this time we'll focus on a high-level keras interface to recurrent neural networks. It is as simple as you can get with RNN, allbeit somewhat constraining for complex tasks like seq2seq.

By default, all keras RNNs apply to a whole sequence of inputs and produce a sequence of hidden states `(return_sequences=True` or just the last hidden state `(return_sequences=False)`. All the recurrence is happening under the hood.

At the top of our model we need to apply a Dense layer to each time-step independently. As of now, by default keras.layers.Dense would apply once to all time-steps concatenated. We use __keras.layers.TimeDistributed__ to modify Dense layer so that it would apply across both batch and time axes.


```python
import keras
import keras.layers as L

model = keras.models.Sequential()
model.add(L.InputLayer([None],dtype='int32'))
model.add(L.Embedding(len(all_words),50))
model.add(L.SimpleRNN(64,return_sequences=True))

#add top layer that predicts tag probabilities
stepwise_dense = L.Dense(len(all_tags),activation='softmax')
stepwise_dense = L.TimeDistributed(stepwise_dense)
model.add(stepwise_dense)
```

    Using TensorFlow backend.


__Training:__ in this case we don't want to prepare the whole training dataset in advance. __The main cause is that the length of every batch depends on the maximum sentence length within the batch.__ This leaves us two options: use custom training code as in previous seminar or use generators.

Keras models have a __`model.fit_generator`__ method that accepts a python generator yielding one batch at a time. But first we need to implement such generator:


```python
from keras.utils.np_utils import to_categorical
BATCH_SIZE=32
def generate_batches(sentences,batch_size=BATCH_SIZE,max_len=None,pad=0):
    assert isinstance(sentences,np.ndarray),"Make sure sentences is q numpy array"
    while True:
        indices = np.random.permutation(np.arange(len(sentences)))
        for start in range(0,len(indices)-1,batch_size):
            batch_indices = indices[start:start+batch_size]
            batch_words,batch_tags = [],[]
            for sent in sentences[batch_indices]:
                words,tags = zip(*sent)
                batch_words.append(words)
                batch_tags.append(tags)

            batch_words = to_matrix(batch_words,word_to_id,max_len,pad)
            batch_tags = to_matrix(batch_tags,tag_to_id,max_len,pad)

            batch_tags_1hot = to_categorical(batch_tags,len(all_tags)).reshape(batch_tags.shape+(-1,))
            yield batch_words,batch_tags_1hot
```

__Callbacks:__ Another thing we need is to measure model performance. The tricky part is not to count accuracy after sentence ends (on padding) and making sure we count all the validation data exactly once.

While it isn't impossible to persuade Keras to do all of that, we may as well write our own callback that does that.
Keras callbacks allow you to write a custom code to be ran once every epoch or every minibatch. We'll define one via LambdaCallback


```python
def compute_test_accuracy(model):
    test_words,test_tags = zip(*[zip(*sentence) for sentence in test_data])
    print(test_words, test_tags)
    test_words,test_tags = to_matrix(test_words,word_to_id),to_matrix(test_tags,tag_to_id)
    print(test_words, test_tags)
    
    #predict tag probabilities of shape [batch,time,n_tags]
    predicted_tag_probabilities = model.predict(test_words,verbose=1)
    predicted_tags = predicted_tag_probabilities.argmax(axis=-1)

    #compute accurary excluding padding
    numerator = np.sum(np.logical_and((predicted_tags == test_tags),(test_words != 0)))
    denominator = np.sum(test_words != 0)
    return float(numerator)/denominator


class EvaluateAccuracy(keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs=None):
        sys.stdout.flush()
        print("\nMeasuring validation accuracy...")
        acc = compute_test_accuracy(self.model)
        print("\nValidation accuracy: %.5f\n"%acc)
        sys.stdout.flush()
        
```


```python
model.compile('adam','categorical_crossentropy')

model.fit_generator(generate_batches(train_data),len(train_data)/BATCH_SIZE,
                    callbacks=[EvaluateAccuracy()], epochs=5,)
```

    Epoch 1/5
    1343/1343 [============================>.] - ETA: 0s - loss: 0.2593
    Measuring validation accuracy...


    IOPub data rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_data_rate_limit`.


    [[ 301  657  432 ...,    0    0    0]
     [   6   41  210 ...,    0    0    0]
     [6000 1101    2 ...,    0    0    0]
     ..., 
     [3282    1 1484 ...,    0    0    0]
     [4863    3   14 ...,    0    0    0]
     [   2  290  176 ...,    0    0    0]] [[13  3  3 ...,  0  0  0]
     [12  5  9 ...,  0  0  0]
     [ 3  9  6 ...,  0  0  0]
     ..., 
     [ 3  3  3 ...,  0  0  0]
     [ 3  7  4 ...,  0  0  0]
     [ 6  3  3 ...,  0  0  0]]
    14304/14335 [============================>.] - ETA: 0s
    Validation accuracy: 0.94000
    
    1344/1343 [==============================] - 98s - loss: 0.2591    
    Epoch 2/5
    1343/1343 [============================>.] - ETA: 0s - loss: 0.0589
    Measuring validation accuracy...


    IOPub data rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_data_rate_limit`.


    [[ 301  657  432 ...,    0    0    0]
     [   6   41  210 ...,    0    0    0]
     [6000 1101    2 ...,    0    0    0]
     ..., 
     [3282    1 1484 ...,    0    0    0]
     [4863    3   14 ...,    0    0    0]
     [   2  290  176 ...,    0    0    0]] [[13  3  3 ...,  0  0  0]
     [12  5  9 ...,  0  0  0]
     [ 3  9  6 ...,  0  0  0]
     ..., 
     [ 3  3  3 ...,  0  0  0]
     [ 3  7  4 ...,  0  0  0]
     [ 6  3  3 ...,  0  0  0]]
    14335/14335 [==============================] - 20s    
    
    Validation accuracy: 0.94422
    
    1344/1343 [==============================] - 99s - loss: 0.0589    
    Epoch 3/5
    1343/1343 [============================>.] - ETA: 0s - loss: 0.0518
    Measuring validation accuracy...


    IOPub data rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_data_rate_limit`.


    [[ 301  657  432 ...,    0    0    0]
     [   6   41  210 ...,    0    0    0]
     [6000 1101    2 ...,    0    0    0]
     ..., 
     [3282    1 1484 ...,    0    0    0]
     [4863    3   14 ...,    0    0    0]
     [   2  290  176 ...,    0    0    0]] [[13  3  3 ...,  0  0  0]
     [12  5  9 ...,  0  0  0]
     [ 3  9  6 ...,  0  0  0]
     ..., 
     [ 3  3  3 ...,  0  0  0]
     [ 3  7  4 ...,  0  0  0]
     [ 6  3  3 ...,  0  0  0]]
    14304/14335 [============================>.] - ETA: 0s
    Validation accuracy: 0.94580
    
    1344/1343 [==============================] - 98s - loss: 0.0518    
    Epoch 4/5
    1343/1343 [============================>.] - ETA: 0s - loss: 0.0470
    Measuring validation accuracy...


    IOPub data rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_data_rate_limit`.


    [[ 301  657  432 ...,    0    0    0]
     [   6   41  210 ...,    0    0    0]
     [6000 1101    2 ...,    0    0    0]
     ..., 
     [3282    1 1484 ...,    0    0    0]
     [4863    3   14 ...,    0    0    0]
     [   2  290  176 ...,    0    0    0]] [[13  3  3 ...,  0  0  0]
     [12  5  9 ...,  0  0  0]
     [ 3  9  6 ...,  0  0  0]
     ..., 
     [ 3  3  3 ...,  0  0  0]
     [ 3  7  4 ...,  0  0  0]
     [ 6  3  3 ...,  0  0  0]]
    14335/14335 [==============================] - 19s    
    
    Validation accuracy: 0.94630
    
    1344/1343 [==============================] - 96s - loss: 0.0470    
    Epoch 5/5
    1343/1343 [============================>.] - ETA: 0s - loss: 0.0428- ETA: 1
    Measuring validation accuracy...


    IOPub data rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_data_rate_limit`.


    [[ 301  657  432 ...,    0    0    0]
     [   6   41  210 ...,    0    0    0]
     [6000 1101    2 ...,    0    0    0]
     ..., 
     [3282    1 1484 ...,    0    0    0]
     [4863    3   14 ...,    0    0    0]
     [   2  290  176 ...,    0    0    0]] [[13  3  3 ...,  0  0  0]
     [12  5  9 ...,  0  0  0]
     [ 3  9  6 ...,  0  0  0]
     ..., 
     [ 3  3  3 ...,  0  0  0]
     [ 3  7  4 ...,  0  0  0]
     [ 6  3  3 ...,  0  0  0]]
    14304/14335 [============================>.] - ETA: 0s
    Validation accuracy: 0.94594
    
    1344/1343 [==============================] - 94s - loss: 0.0428    





    <keras.callbacks.History at 0x7f80888485f8>



Measure final accuracy on the whole test set.


```python
acc = compute_test_accuracy(model)
print("Final accuracy: %.5f"%acc)

assert acc>0.94, "Keras has gone on a rampage again, please contact course staff."
```

    IOPub data rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_data_rate_limit`.


    [[ 301  657  432 ...,    0    0    0]
     [   6   41  210 ...,    0    0    0]
     [6000 1101    2 ...,    0    0    0]
     ..., 
     [3282    1 1484 ...,    0    0    0]
     [4863    3   14 ...,    0    0    0]
     [   2  290  176 ...,    0    0    0]] [[13  3  3 ...,  0  0  0]
     [12  5  9 ...,  0  0  0]
     [ 3  9  6 ...,  0  0  0]
     ..., 
     [ 3  3  3 ...,  0  0  0]
     [ 3  7  4 ...,  0  0  0]
     [ 6  3  3 ...,  0  0  0]]
    14335/14335 [==============================] - 19s    
    Final accuracy: 0.94594


### Task I: getting all bidirectional

Since we're analyzing a full sequence, it's legal for us to look into future data.

A simple way to achieve that is to go both directions at once, making a __bidirectional RNN__.

In Keras you can achieve that both manually (using two LSTMs and Concatenate) and by using __`keras.layers.Bidirectional`__. 

This one works just as `TimeDistributed` we saw before: you wrap it around a recurrent layer (SimpleRNN now and LSTM/GRU later) and it actually creates two layers under the hood.

Your first task is to use such a layer our POS-tagger.


```python
#Define a model that utilizes bidirectional SimpleRNN
model = keras.models.Sequential()

model.add(L.InputLayer([None],dtype='int32'))
model.add(L.Embedding(len(all_words),50))
model.add(L.Bidirectional(L.SimpleRNN(64,return_sequences=True)))

#add top layer that predicts tag probabilities
stepwise_dense = L.Dense(len(all_tags),activation='softmax')
stepwise_dense = L.TimeDistributed(stepwise_dense)
model.add(stepwise_dense)
```


```python
model.compile('adam','categorical_crossentropy')

model.fit_generator(generate_batches(train_data),len(train_data)/BATCH_SIZE,
                    callbacks=[EvaluateAccuracy()], epochs=5,)
```

    Epoch 1/5
    1343/1343 [============================>.] - ETA: 0s - loss: 0.1878- ETA: 1s
    Measuring validation accuracy...


    IOPub data rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_data_rate_limit`.


    [[ 301  657  432 ...,    0    0    0]
     [   6   41  210 ...,    0    0    0]
     [6000 1101    2 ...,    0    0    0]
     ..., 
     [3282    1 1484 ...,    0    0    0]
     [4863    3   14 ...,    0    0    0]
     [   2  290  176 ...,    0    0    0]] [[13  3  3 ...,  0  0  0]
     [12  5  9 ...,  0  0  0]
     [ 3  9  6 ...,  0  0  0]
     ..., 
     [ 3  3  3 ...,  0  0  0]
     [ 3  7  4 ...,  0  0  0]
     [ 6  3  3 ...,  0  0  0]]
    14335/14335 [==============================] - 35s    
    
    Validation accuracy: 0.95584
    
    1344/1343 [==============================] - 154s - loss: 0.1877   
    Epoch 2/5
    1343/1343 [============================>.] - ETA: 0s - loss: 0.0426
    Measuring validation accuracy...


    IOPub data rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_data_rate_limit`.


    [[ 301  657  432 ...,    0    0    0]
     [   6   41  210 ...,    0    0    0]
     [6000 1101    2 ...,    0    0    0]
     ..., 
     [3282    1 1484 ...,    0    0    0]
     [4863    3   14 ...,    0    0    0]
     [   2  290  176 ...,    0    0    0]] [[13  3  3 ...,  0  0  0]
     [12  5  9 ...,  0  0  0]
     [ 3  9  6 ...,  0  0  0]
     ..., 
     [ 3  3  3 ...,  0  0  0]
     [ 3  7  4 ...,  0  0  0]
     [ 6  3  3 ...,  0  0  0]]
    14335/14335 [==============================] - 36s    
    
    Validation accuracy: 0.96086
    
    1344/1343 [==============================] - 152s - loss: 0.0426   
    Epoch 3/5
    1343/1343 [============================>.] - ETA: 0s - loss: 0.0352
    Measuring validation accuracy...


    IOPub data rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_data_rate_limit`.


    [[ 301  657  432 ...,    0    0    0]
     [   6   41  210 ...,    0    0    0]
     [6000 1101    2 ...,    0    0    0]
     ..., 
     [3282    1 1484 ...,    0    0    0]
     [4863    3   14 ...,    0    0    0]
     [   2  290  176 ...,    0    0    0]] [[13  3  3 ...,  0  0  0]
     [12  5  9 ...,  0  0  0]
     [ 3  9  6 ...,  0  0  0]
     ..., 
     [ 3  3  3 ...,  0  0  0]
     [ 3  7  4 ...,  0  0  0]
     [ 6  3  3 ...,  0  0  0]]
    14335/14335 [==============================] - 35s    
    
    Validation accuracy: 0.96183
    
    1344/1343 [==============================] - 151s - loss: 0.0352   
    Epoch 4/5
    1343/1343 [============================>.] - ETA: 0s - loss: 0.0300
    Measuring validation accuracy...


    IOPub data rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_data_rate_limit`.


    [[ 301  657  432 ...,    0    0    0]
     [   6   41  210 ...,    0    0    0]
     [6000 1101    2 ...,    0    0    0]
     ..., 
     [3282    1 1484 ...,    0    0    0]
     [4863    3   14 ...,    0    0    0]
     [   2  290  176 ...,    0    0    0]] [[13  3  3 ...,  0  0  0]
     [12  5  9 ...,  0  0  0]
     [ 3  9  6 ...,  0  0  0]
     ..., 
     [ 3  3  3 ...,  0  0  0]
     [ 3  7  4 ...,  0  0  0]
     [ 6  3  3 ...,  0  0  0]]
    14335/14335 [==============================] - 36s    
    
    Validation accuracy: 0.96244
    
    1344/1343 [==============================] - 153s - loss: 0.0300   
    Epoch 5/5
    1343/1343 [============================>.] - ETA: 0s - loss: 0.0253
    Measuring validation accuracy...


    IOPub data rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_data_rate_limit`.


    [[ 301  657  432 ...,    0    0    0]
     [   6   41  210 ...,    0    0    0]
     [6000 1101    2 ...,    0    0    0]
     ..., 
     [3282    1 1484 ...,    0    0    0]
     [4863    3   14 ...,    0    0    0]
     [   2  290  176 ...,    0    0    0]] [[13  3  3 ...,  0  0  0]
     [12  5  9 ...,  0  0  0]
     [ 3  9  6 ...,  0  0  0]
     ..., 
     [ 3  3  3 ...,  0  0  0]
     [ 3  7  4 ...,  0  0  0]
     [ 6  3  3 ...,  0  0  0]]
    14335/14335 [==============================] - 36s    
    
    Validation accuracy: 0.96186
    
    1344/1343 [==============================] - 154s - loss: 0.0253   





    <keras.callbacks.History at 0x7f802c338908>




```python
acc = compute_test_accuracy(model)
print("\nFinal accuracy: %.5f"%acc)

assert acc>0.96, "Bidirectional RNNs are better than this!"
print("Well done!")
```

    IOPub data rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_data_rate_limit`.


    [[ 301  657  432 ...,    0    0    0]
     [   6   41  210 ...,    0    0    0]
     [6000 1101    2 ...,    0    0    0]
     ..., 
     [3282    1 1484 ...,    0    0    0]
     [4863    3   14 ...,    0    0    0]
     [   2  290  176 ...,    0    0    0]] [[13  3  3 ...,  0  0  0]
     [12  5  9 ...,  0  0  0]
     [ 3  9  6 ...,  0  0  0]
     ..., 
     [ 3  3  3 ...,  0  0  0]
     [ 3  7  4 ...,  0  0  0]
     [ 6  3  3 ...,  0  0  0]]
    14335/14335 [==============================] - 36s    
    
    Final accuracy: 0.96186
    Well done!


### Task II: now go and improve it

You guesses it. We're now gonna ask you to come up with a better network.

Here's a few tips:

* __Go beyond SimpleRNN__: there's `keras.layers.LSTM` and `keras.layers.GRU`
  * If you want to use a custom recurrent Cell, read [this](https://keras.io/layers/recurrent/#rnn)
  * You can also use 1D Convolutions (`keras.layers.Conv1D`). They are often as good as recurrent layers but with less overfitting.
* __Stack more layers__: if there is a common motif to this course it's about stacking layers
  * You can just add recurrent and 1dconv layers on top of one another and keras will understand it
  * Just remember that bigger networks may need more epochs to train
* __Gradient clipping__: If your training isn't as stable as you'd like, set `clipnorm` in your optimizer.
  * Which is to say, it's a good idea to watch over your loss curve at each minibatch. Try tensorboard callback or something similar.
* __Regularization__: you can apply dropouts as usuall but also in an RNN-specific way
  * `keras.layers.Dropout` works inbetween RNN layers
  * Recurrent layers also have `recurrent_dropout` parameter
* __More words!__: You can obtain greater performance by expanding your model's input dictionary from 5000 to up to every single word!
  * Just make sure your model doesn't overfit due to so many parameters.
  * Combined with regularizers or pre-trained word-vectors this could be really good cuz right now our model is blind to >5% of words.
* __The most important advice__: don't cram in everything at once!
  * If you stuff in a lot of modiffications, some of them almost inevitably gonna be detrimental and you'll never know which of them are.
  * Try to instead go in small iterations and record experiment results to guide further search.
  
There's some advanced stuff waiting at the end of the notebook.
  
Good hunting!


```python
#Define a model that utilizes bidirectional SimpleRNN
model = keras.models.Sequential()
model.add(L.InputLayer([None],dtype='int32'))
model.add(L.Embedding(len(all_words),50))
model.add(L.Bidirectional(L.GRU(128,return_sequences=True,activation='relu')))
model.add(L.Dropout(0.5))
model.add(L.Bidirectional(L.GRU(64,return_sequences=True,activation='relu')))
model.add(L.Dropout(0.5))

#add top layer that predicts tag probabilities
stepwise_dense = L.Dense(len(all_tags),activation='softmax')
stepwise_dense = L.TimeDistributed(stepwise_dense)
model.add(stepwise_dense)
```


```python
#feel free to change anything here

model.compile('adam','categorical_crossentropy')

model.fit_generator(generate_batches(train_data),len(train_data)/BATCH_SIZE,
                    callbacks=[EvaluateAccuracy()], epochs=5,)
```

    Epoch 1/5
    1343/1343 [============================>.] - ETA: 0s - loss: 0.1956
    Measuring validation accuracy...


    IOPub data rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_data_rate_limit`.


    [[ 301  657  432 ...,    0    0    0]
     [   6   41  210 ...,    0    0    0]
     [6000 1101    2 ...,    0    0    0]
     ..., 
     [3282    1 1484 ...,    0    0    0]
     [4863    3   14 ...,    0    0    0]
     [   2  290  176 ...,    0    0    0]] [[13  3  3 ...,  0  0  0]
     [12  5  9 ...,  0  0  0]
     [ 3  9  6 ...,  0  0  0]
     ..., 
     [ 3  3  3 ...,  0  0  0]
     [ 3  7  4 ...,  0  0  0]
     [ 6  3  3 ...,  0  0  0]]
    14335/14335 [==============================] - 231s   
    
    Validation accuracy: 0.95648
    
    1344/1343 [==============================] - 1030s - loss: 0.1955  
    Epoch 2/5
     439/1343 [========>.....................] - ETA: 521s - loss: 0.0568


```python
acc = compute_test_accuracy(model)
print("\nFinal accuracy: %.5f"%acc)

if acc >= 0.99:
    print("Awesome! Sky was the limit and yet you scored even higher!")
elif acc >= 0.98:
    print("Excellent! Whatever dark magic you used, it certainly did it's trick.")
elif acc >= 0.97:
    print("Well done! If this was a graded assignment, you would have gotten a 100% score.")
elif acc > 0.96:
    print("Just a few more iterations!")
else:
    print("There seems to be something broken in the model. Unless you know what you're doing, try taking bidirectional RNN and adding one enhancement at a time to see where's the problem.")
```


#### Some advanced stuff
Here there are a few more tips on how to improve training that are a bit trickier to impliment. We strongly suggest that you try them _after_ you've got a good initial model.
* __Use pre-trained embeddings__: you can use pre-trained weights from [there](http://ahogrammer.com/2017/01/20/the-list-of-pretrained-word-embeddings/) to kickstart your Embedding layer.
  * Embedding layer has a matrix W (layer.W) which contains word embeddings for each word in the dictionary. You can just overwrite them with tf.assign.
  * When using pre-trained embeddings, pay attention to the fact that model's dictionary is different from your own.
  * You may want to switch trainable=False for embedding layer in first few epochs as in regular fine-tuning.  
* __More efficient baching__: right now TF spends a lot of time iterating over "0"s
  * This happens because batch is always padded to the length of a longest sentence
  * You can speed things up by pre-generating batches of similar lengths and feeding it with randomly chosen pre-generated batch.
  * This technically breaks the i.i.d. assumption, but it works unless you come up with some insane rnn architectures.
* __Structured loss functions__: since we're tagging the whole sequence at once, we might as well train our network to do so.
  * There's more than one way to do so, but we'd recommend starting with [Conditional Random Fields](http://blog.echen.me/2012/01/03/introduction-to-conditional-random-fields/)
  * You could plug CRF as a loss function and still train by backprop. There's even some neat tensorflow [implementation](https://www.tensorflow.org/api_guides/python/contrib.crf) for you.

