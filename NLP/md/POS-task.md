
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

    [nltk_data] Downloading package brown to /home/karen/nltk_data...
    [nltk_data]   Unzipping corpora/brown.zip.
    [nltk_data] Downloading package universal_tagset to
    [nltk_data]     /home/karen/nltk_data...
    [nltk_data]   Unzipping taggers/universal_tagset.zip.



```python
data
```




    array([list([('the', 'DET'), ('fulton', 'NOUN'), ('county', 'NOUN'), ('grand', 'ADJ'), ('jury', 'NOUN'), ('said', 'VERB'), ('friday', 'NOUN'), ('an', 'DET'), ('investigation', 'NOUN'), ('of', 'ADP'), ("atlanta's", 'NOUN'), ('recent', 'ADJ'), ('primary', 'NOUN'), ('election', 'NOUN'), ('produced', 'VERB'), ('``', '.'), ('no', 'DET'), ('evidence', 'NOUN'), ("''", '.'), ('that', 'ADP'), ('any', 'DET'), ('irregularities', 'NOUN'), ('took', 'VERB'), ('place', 'NOUN'), ('.', '.')]),
           list([('the', 'DET'), ('jury', 'NOUN'), ('further', 'ADV'), ('said', 'VERB'), ('in', 'ADP'), ('term-end', 'NOUN'), ('presentments', 'NOUN'), ('that', 'ADP'), ('the', 'DET'), ('city', 'NOUN'), ('executive', 'ADJ'), ('committee', 'NOUN'), (',', '.'), ('which', 'DET'), ('had', 'VERB'), ('over-all', 'ADJ'), ('charge', 'NOUN'), ('of', 'ADP'), ('the', 'DET'), ('election', 'NOUN'), (',', '.'), ('``', '.'), ('deserves', 'VERB'), ('the', 'DET'), ('praise', 'NOUN'), ('and', 'CONJ'), ('thanks', 'NOUN'), ('of', 'ADP'), ('the', 'DET'), ('city', 'NOUN'), ('of', 'ADP'), ('atlanta', 'NOUN'), ("''", '.'), ('for', 'ADP'), ('the', 'DET'), ('manner', 'NOUN'), ('in', 'ADP'), ('which', 'DET'), ('the', 'DET'), ('election', 'NOUN'), ('was', 'VERB'), ('conducted', 'VERB'), ('.', '.')]),
           list([('the', 'DET'), ('september-october', 'NOUN'), ('term', 'NOUN'), ('jury', 'NOUN'), ('had', 'VERB'), ('been', 'VERB'), ('charged', 'VERB'), ('by', 'ADP'), ('fulton', 'NOUN'), ('superior', 'ADJ'), ('court', 'NOUN'), ('judge', 'NOUN'), ('durwood', 'NOUN'), ('pye', 'NOUN'), ('to', 'PRT'), ('investigate', 'VERB'), ('reports', 'NOUN'), ('of', 'ADP'), ('possible', 'ADJ'), ('``', '.'), ('irregularities', 'NOUN'), ("''", '.'), ('in', 'ADP'), ('the', 'DET'), ('hard-fought', 'ADJ'), ('primary', 'NOUN'), ('which', 'DET'), ('was', 'VERB'), ('won', 'VERB'), ('by', 'ADP'), ('mayor-nominate', 'NOUN'), ('ivan', 'NOUN'), ('allen', 'NOUN'), ('jr.', 'NOUN'), ('.', '.')]),
           ...,
           list([('the', 'DET'), ('doors', 'NOUN'), ('of', 'ADP'), ('the', 'DET'), ('d', 'NOUN'), ('train', 'NOUN'), ('slid', 'VERB'), ('shut', 'VERB'), (',', '.'), ('and', 'CONJ'), ('as', 'ADP'), ('i', 'PRON'), ('dropped', 'VERB'), ('into', 'ADP'), ('a', 'DET'), ('seat', 'NOUN'), ('and', 'CONJ'), (',', '.'), ('exhaling', 'VERB'), (',', '.'), ('looked', 'VERB'), ('up', 'PRT'), ('across', 'ADP'), ('the', 'DET'), ('aisle', 'NOUN'), (',', '.'), ('the', 'DET'), ('whole', 'ADJ'), ('aviary', 'NOUN'), ('in', 'ADP'), ('my', 'DET'), ('head', 'NOUN'), ('burst', 'VERB'), ('into', 'ADP'), ('song', 'NOUN'), ('.', '.')]),
           list([('she', 'PRON'), ('was', 'VERB'), ('a', 'DET'), ('living', 'VERB'), ('doll', 'NOUN'), ('and', 'CONJ'), ('no', 'DET'), ('mistake', 'NOUN'), ('--', '.'), ('the', 'DET'), ('blue-black', 'ADJ'), ('bang', 'NOUN'), (',', '.'), ('the', 'DET'), ('wide', 'ADJ'), ('cheekbones', 'NOUN'), (',', '.'), ('olive-flushed', 'ADJ'), (',', '.'), ('that', 'PRON'), ('betrayed', 'VERB'), ('the', 'DET'), ('cherokee', 'NOUN'), ('strain', 'NOUN'), ('in', 'ADP'), ('her', 'DET'), ('midwestern', 'ADJ'), ('lineage', 'NOUN'), (',', '.'), ('and', 'CONJ'), ('the', 'DET'), ('mouth', 'NOUN'), ('whose', 'DET'), ('only', 'ADJ'), ('fault', 'NOUN'), (',', '.'), ('in', 'ADP'), ('the', 'DET'), ("novelist's", 'NOUN'), ('carping', 'VERB'), ('phrase', 'NOUN'), (',', '.'), ('was', 'VERB'), ('that', 'ADP'), ('the', 'DET'), ('lower', 'ADJ'), ('lip', 'NOUN'), ('was', 'VERB'), ('a', 'DET'), ('trifle', 'NOUN'), ('too', 'ADV'), ('voluptuous', 'ADJ'), ('.', '.')]),
           list([('from', 'ADP'), ('what', 'DET'), ('i', 'PRON'), ('was', 'VERB'), ('able', 'ADJ'), ('to', 'ADP'), ('gauge', 'NOUN'), ('in', 'ADP'), ('a', 'DET'), ('swift', 'ADJ'), (',', '.'), ('greedy', 'ADJ'), ('glance', 'NOUN'), (',', '.'), ('the', 'DET'), ('figure', 'NOUN'), ('inside', 'ADP'), ('the', 'DET'), ('coral-colored', 'ADJ'), ('boucle', 'NOUN'), ('dress', 'NOUN'), ('was', 'VERB'), ('stupefying', 'VERB'), ('.', '.')])],
          dtype=object)




```python
from sklearn.model_selection import train_test_split
train_data,test_data = train_test_split(data,test_size=0.25,random_state=42)
```


```python
train_data.shape, test_data.shape
```




    ((43005,), (14335,))




```python
from IPython.display import HTML, display
def draw(sentence):
    words,tags = zip(*sentence)
    display(HTML('<table><tr>{tags}</tr>{words}<tr></table>'.format(
                words = '<td>{}</td>'.format('</td><td>'.join(words)),
                tags = '<td>{}</td>'.format('</td><td>'.join(tags)))))
    
    
draw(data[11])
draw(data[10])
draw(data[7])
```


<table><tr><td>NOUN</td><td>ADP</td><td>NOUN</td><td>NOUN</td><td>NOUN</td><td>NOUN</td><td>VERB</td><td>ADV</td><td>VERB</td><td>ADP</td><td>DET</td><td>ADJ</td><td>NOUN</td><td>.</td></tr><td>implementation</td><td>of</td><td>georgia's</td><td>automobile</td><td>title</td><td>law</td><td>was</td><td>also</td><td>recommended</td><td>by</td><td>the</td><td>outgoing</td><td>jury</td><td>.</td><tr></table>



<table><tr><td>PRON</td><td>VERB</td><td>ADP</td><td>DET</td><td>NOUN</td><td>.</td><td>VERB</td><td>NOUN</td><td>PRT</td><td>VERB</td><td>.</td><td>DET</td><td>NOUN</td><td>.</td></tr><td>it</td><td>urged</td><td>that</td><td>the</td><td>city</td><td>``</td><td>take</td><td>steps</td><td>to</td><td>remedy</td><td>''</td><td>this</td><td>problem</td><td>.</td><tr></table>



<table><tr><td>NOUN</td><td>VERB</td></tr><td>merger</td><td>proposed</td><tr></table>


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




    Counter({'122': 2,
             'shrill': 7,
             'rhine-westphalia': 1,
             'reguli': 1,
             'otter': 5,
             'airline': 2,
             "painter's": 3,
             'battleground': 2,
             'uncousinly': 1,
             'invigoration': 2,
             'bruce': 4,
             'lasso': 2,
             'gute': 2,
             'incipient': 4,
             'druggan-lake': 1,
             'symbolists': 1,
             'glen': 7,
             'expresses': 9,
             'infirm': 1,
             'bundle': 20,
             'ranking': 5,
             'ignoramus': 2,
             'screwed': 14,
             'outback': 3,
             'glycols': 1,
             "vermont's": 2,
             'improvised': 3,
             'loveliest': 3,
             'loft': 2,
             'continuance': 6,
             'breakables': 1,
             'conducive': 2,
             'doves': 1,
             'indecision': 5,
             'dinners': 9,
             'juleps': 1,
             'consign': 2,
             'yrs.': 4,
             'degassed': 1,
             'sake': 41,
             'alternatively': 3,
             'handstands': 3,
             'glistening': 6,
             "mahzeer's": 5,
             'durable': 12,
             'bark': 14,
             'moderating': 1,
             "didn't": 401,
             'mondays': 1,
             'rotunda': 6,
             'gen.': 23,
             'series': 130,
             'marksmanship': 4,
             'boeotian': 1,
             'vittorio': 2,
             'inter-tribal': 1,
             'beans': 9,
             'erupt': 2,
             'expansiveness': 3,
             'bonheur': 1,
             'hardy': 42,
             'physicists': 2,
             'resting': 19,
             'livshitz': 2,
             'extraterrestrial': 3,
             'tenses': 1,
             'composites': 1,
             'decks': 6,
             'sundry': 5,
             'old-time': 4,
             'uncomfortably': 3,
             'unimportant': 9,
             'motets': 1,
             'anglican': 11,
             "can't": 169,
             'nepal': 1,
             'danzig': 1,
             'thrashed': 3,
             'altruism': 1,
             'differentiated': 5,
             'timeless': 2,
             "wert's": 1,
             'growling': 1,
             'lambeth': 3,
             'yum-yum': 1,
             'plee-zing': 1,
             '11.2': 1,
             'ajb': 1,
             'pleads': 1,
             'abdomen': 6,
             'singularly': 1,
             'batavia': 3,
             'a.m.': 40,
             'fountainhead': 1,
             'best-gaited': 1,
             'intelligible': 11,
             'enters': 13,
             'hinckley': 1,
             'stoic': 3,
             'mich.': 4,
             'alibi': 8,
             'trumps': 1,
             "shafer's": 1,
             'morphemic': 1,
             'tiniest': 3,
             'pinhead': 1,
             'crystallization': 3,
             'saturday': 67,
             'trousers-pockets': 1,
             'assiniboia': 2,
             'recurrently': 1,
             'imperious': 1,
             'cloakrooms': 1,
             'snobbery': 4,
             'weight-height': 1,
             '8,100': 1,
             'bubenik': 1,
             'indexes': 2,
             'drag': 15,
             'bernz-o-matic': 1,
             'shipyards': 1,
             'reflection': 32,
             '37,470': 1,
             'myers': 3,
             "quake's": 1,
             'analogues': 1,
             'steele': 21,
             'urine': 1,
             '5031': 1,
             'chew': 2,
             'federalism': 2,
             'tying': 5,
             'highfield': 1,
             'cowbirds': 3,
             'kika': 1,
             'inter': 2,
             'overtones': 4,
             'halfback': 10,
             '1,450,000': 1,
             'combat-inflicted': 1,
             'affirmations': 1,
             'paycheck': 2,
             'moldavian': 1,
             'mimieux': 1,
             'great': 665,
             'pacem': 1,
             'winging': 1,
             'telomeric': 1,
             'headwalls': 2,
             'gallup': 1,
             'star-spangled': 1,
             'rotten': 2,
             'unleveled': 1,
             'snags': 1,
             'bop': 3,
             'grossman': 1,
             'one-eighth': 2,
             'glittered': 1,
             'mullen': 2,
             'hawkins': 2,
             "rogues'": 1,
             'interact': 2,
             '75-minute': 1,
             'slaves': 44,
             'appearing': 16,
             'aggressively': 2,
             'gaucherie': 1,
             'brett': 4,
             'avocado': 10,
             'assaults': 4,
             'expressway': 10,
             'work': 762,
             'squandered': 2,
             'psychotherapists': 1,
             'recordings': 11,
             'strung': 4,
             "where're": 1,
             'reckoned': 3,
             'evade': 1,
             'breakthrough': 6,
             'kolpakova': 2,
             'bayonet': 6,
             'kissed': 15,
             'prefaced': 2,
             'endorse': 6,
             'femme': 1,
             'bathtubs': 1,
             'autistic': 13,
             'sojourners': 1,
             'yardage': 2,
             'isis': 1,
             'abstractive': 1,
             'imperfection': 1,
             'maht': 1,
             'freeholder': 1,
             'evolutionists': 1,
             'bachelor-type': 1,
             'quemoy': 2,
             'dams': 3,
             'mentioned': 79,
             'divided': 55,
             '72': 10,
             'hockey': 1,
             'gainful': 1,
             'interspecies': 1,
             'sundown': 6,
             'bright': 87,
             'seagoville': 1,
             'cologne': 9,
             'trained': 54,
             'tremendously': 10,
             'pectorals': 1,
             'clicked': 8,
             'tightly': 15,
             'unbalanced': 3,
             'upshots': 1,
             'preconscious': 1,
             'reflectance': 1,
             'neal': 4,
             'rabaul': 1,
             'seventy-four': 1,
             "stanley's": 4,
             'spahn': 5,
             'fermentation': 3,
             'tizard': 1,
             'basic': 171,
             'accelerometer': 17,
             'warms': 1,
             'cadillac': 9,
             'densities': 2,
             'stooooomp': 1,
             'lifeboat': 4,
             'indisposed': 3,
             '271': 2,
             'photocathodes': 2,
             'decrees': 5,
             'debilitating': 2,
             'northward': 5,
             'say': 504,
             'litigants': 3,
             'calisthenics': 4,
             'ego': 13,
             'ex-president': 2,
             "pianist's": 4,
             'unmatched': 2,
             'turandot': 1,
             'seven-thirty': 1,
             'bettering': 1,
             'rayburn-johnson': 1,
             '47.1%': 1,
             'overpayment': 4,
             'gardeners': 1,
             'soft-spoken': 1,
             'threshold': 15,
             'claiming': 16,
             'drunkard': 3,
             'modifying': 4,
             'chances': 24,
             'geary': 1,
             'upi': 5,
             'trauma': 1,
             'unite': 10,
             'fugual': 1,
             'squares': 13,
             '$3.5': 2,
             'non-supervisory': 3,
             'benefit': 63,
             'pyrex': 5,
             'compartment': 11,
             'cross-sectional': 4,
             'co-signers': 1,
             'majestic': 10,
             'thinner': 6,
             'unreconstructed': 5,
             'banal': 2,
             '695': 1,
             'megawatt': 2,
             'barre-montpelier': 1,
             'sochi': 2,
             'off-beat': 2,
             'dismissal': 7,
             'vowels': 3,
             'worth-waiting-for': 1,
             'irene': 2,
             'tardiness': 1,
             'feasible': 15,
             'moans': 1,
             'eye-gouging': 1,
             'navels': 1,
             'thills': 1,
             'skirt': 21,
             'entries': 19,
             'strafaci': 1,
             'chives': 1,
             'non-success': 1,
             'fairly': 58,
             'arrowheads': 1,
             'professedly': 3,
             'obe': 1,
             'corning': 1,
             'extricate': 2,
             'patina': 1,
             'lower': 123,
             'shaking': 21,
             'spoken': 37,
             'onslaughts': 2,
             'undaunted': 1,
             'patchen': 26,
             'metronome': 3,
             'minaces': 1,
             'caper': 6,
             "catcher's": 1,
             'zabel': 1,
             'inviolability': 1,
             'outlandish': 1,
             'sessions': 26,
             'bon': 2,
             'temperance': 1,
             'catastrophically': 2,
             'unexpected': 23,
             'boasts': 2,
             'florican-my': 1,
             'finn': 1,
             'wedded': 4,
             'libyan': 2,
             'yourself': 67,
             'drunkenness': 4,
             'hester': 3,
             'paddle': 1,
             'pauson': 2,
             "henley's": 2,
             'inheritors': 1,
             'khrush': 1,
             '$300,000,000': 1,
             'dreadfully': 1,
             '1565': 2,
             'speakership': 1,
             'diversionary': 1,
             'preening': 1,
             'bystander': 1,
             'resumed': 23,
             'suitably': 3,
             'prewar': 1,
             'statue': 17,
             'convinced': 50,
             'stab': 3,
             'stickman': 1,
             'hodge-podge': 1,
             'morbid': 1,
             'sanguineum': 1,
             'child-bearing': 1,
             'beef': 31,
             'tall-growing': 1,
             'inland': 4,
             'loop': 21,
             'overwhelm': 1,
             'participating': 15,
             'flounders': 1,
             'obscenities': 2,
             'notebook': 2,
             "y'r": 1,
             'submissive': 4,
             'parliamentarians': 1,
             'itself': 304,
             '24-inch': 1,
             'jails': 3,
             'religions': 18,
             'coloration': 2,
             'oregonians': 1,
             'uncritically': 1,
             'computations': 1,
             'clauses': 4,
             'mos.': 1,
             'optimal': 28,
             'lugged': 5,
             'shamefacedly': 1,
             'nakamura': 3,
             'china': 69,
             'layoffs': 1,
             'core': 43,
             "soloists'": 1,
             'lovers': 10,
             'mayor-elect': 1,
             'sided': 1,
             'errand': 7,
             "bellamy's": 1,
             'eidetic': 1,
             'fluid-filled': 1,
             'occasionally': 48,
             'loud-voiced': 1,
             'sour': 3,
             'sing-song': 1,
             'ferns': 1,
             'verb': 4,
             '11-5': 1,
             'circuit': 23,
             'travelled': 4,
             'frog': 1,
             'buckhannon': 1,
             'breathed': 9,
             'installations': 16,
             'simply': 171,
             'irate': 1,
             "santa's": 1,
             '360,000': 1,
             'undesirable': 10,
             'pasting': 1,
             'corona': 1,
             'evocations': 2,
             'foreheads': 2,
             'may': 1402,
             'naturally': 70,
             'authorit




<b>limit_output extension: Maximum message size of 10000 exceeded with 24120 characters</b>



```python
from collections import defaultdict
word_to_id = defaultdict(lambda:1,{word:i for i,word in enumerate(all_words)})
tag_to_id = {tag:i for i,tag in enumerate(all_tags)}
```


```python
word_to_id
```




    defaultdict(<function __main__.<lambda>>,
                {'appeals': 5188,
                 'couple': 867,
                 'gavin': 5925,
                 'nuts': 4758,
                 'shelf': 7242,
                 'jig': 9611,
                 'bundle': 4808,
                 'played': 1033,
                 'history': 334,
                 'screwed': 6307,
                 'hollow': 7241,
                 'garibaldi': 6930,
                 'warfare': 2584,
                 'appearing': 5848,
                 'lublin': 6087,
                 'barrels': 9613,
                 'dinners': 8507,
                 'down': 113,
                 'sake': 2687,
                 'violent': 3303,
                 'awake': 4916,
                 'streetcar': 6829,
                 'durable': 7023,
                 'bark': 6308,
                 'news': 1058,
                 "didn't": 227,
                 'dialect': 8368,
                 'gen.': 4297,
                 'series': 797,
                 'top': 476,
                 'solving': 9583,
                 'american': 173,
                 'filled': 1092,
                 'excitement': 3395,
                 'experts': 2905,
                 'catcher': 5576,
                 'planning': 811,
                 'resting': 5013,
                 'beef': 3477,
                 'layer': 7350,
                 'widely': 2156,
                 'children': 278,
                 'unimportant': 8509,
                 'processed': 7349,
                 'minneapolis': 8670,
                 'anglican': 7450,
                 'no': 60,
                 'electronics': 3428,
                 'licked': 8369,
                 'notify': 9816,
                 'probably': 372,
                 'park': 1157,
                 'curriculum': 5849,
                 'peripheral': 9614,
                 'comfortably': 7351,
                 'shear': 2803,
                 'her': 43,
                 'organs': 6461,
                 'a.m.': 2745,
                 'intelligible': 7452,
                 'generously': 9615,
                 'q': 2411,
                 'pitched': 9919,
                 'carriage': 7697,
                 'monday': 1612,
                 'alibi': 9201,
                 'equality': 7352,
                 'seized': 4230,
                 'armed': 1862,
                 'joint': 2865,
                 'saturday': 1635,
                 'present': 251,
                 'pressures': 2928,
                 'diet': 4627,
                 'damn': 3389,
                 'buck': 4859,
                 'calling': 2500,
                 'relied': 9817,
                 'mutual': 3984,
                 'sipping': 9818,
                 'guessing': 9612,
                 'nixon': 4152,
                 'drag': 6005,
                 'reflection': 3354,
                 'bursting': 6964,
                 'pain': 1230,
                 'pushed': 2144,
                 'steele': 4612,
                 'ernest': 8089,
                 'drunk': 3008,
                 'abel': 4962,
                 'encountered': 3575,
                 'listener': 8090,
                 'sanctuary': 9026,
                 'tank': 7142,
                 'league': 1584,
                 'creative': 2280,
                 'alley': 9819,
                 'halfback': 7939,
                 'great': 151,
                 'energies': 7583,
                 'security': 1191,
                 'artificial': 5543,
                 'plates': 4377,
                 'farmhouse': 9866,
                 'glimpse': 5764,
                 'assemble': 8747,
                 'slaves': 2509,
                 'drug': 4268,
                 'empty': 1722,
                 'misunderstanding': 7806,
                 'yelled': 4488,
                 'reflecting': 5496,
                 "you'd": 3046,
                 'comment': 2662,
                 'expressway': 7942,
                 'work': 133,
                 'recordings': 7453,
                 'between': 137,
                 'preparing': 4574,
                 'gonzales': 8671,
                 'disagreement': 7584,
                 'specified': 3763,
                 'lot': 824,
                 'comparing': 8843,
                 'portion': 1800,
                 'contrasting': 7658,
                 'truly': 1974,
                 'pathology': 3302,
                 'tries': 6785,
                 'express': 2677,
                 'background': 1645,
                 'certain': 306,
                 'greeted': 4917,
                 'thunder': 6378,
                 'cumulative': 6931,
                 'divided': 2029,
                 'stumbling': 9634,
                 'pirates': 7143,
                 '72': 7943,
                 'framing': 8370,
                 'no.': 1750,
                 'underneath': 7704,
                 'brothers': 2717,
                 'tolerant': 8861,
                 'destined': 8844,
                 'alter': 6022,
                 'bright': 1236,
                 'southern': 758,
                 'exports': 7585,
                 'trained': 2074,
                 'ideas': 716,
                 'inevitable': 3284,
                 'jr.': 1447,
                 'thereto': 7661,
                 'sponsors': 8092,
                 'tightly': 6008,
                 'relatively': 1275,
                 'budget': 1892,
                 'desert': 4661,
                 'disposal': 4918,
                 'hawk': 6379,
                 'picasso': 6469,
                 'basic': 590,
                 'accelerometer': 5434,
                 'edition': 2973,
                 'cadillac': 8511,
                 'championship': 9637,
                 'prone': 6553,
                 'constructive': 6088,
                 'eventually': 2172,
                 't.': 3854,
                 'discovered': 1489,
                 'developing': 2157,
                 'curse': 7698,
                 'assure': 3009,
                 'say': 183,
                 'haste': 9028,
                 'alarm': 5858,
                 'argued': 3671,
                 'ego': 6630,
                 'planetary': 4662,
                 'profound': 3902,
                 'prospect': 4084,
                 'ordinary': 1521,
                 'patrons': 8668,
                 'threshold': 6009,
                 'science': 794,
                 'claiming': 5686,
                 'hidden': 4919,
                 'chances': 4181,
                 'unite': 7945,
                 'disclosed': 6380,
                 'read': 570,
                 'comic': 8674,
                 'royal': 2309,
                 'precisely': 2310,
                 'benefit': 1742,
                 'compartment': 7454,
                 'due': 727,
                 'inevitably': 2933,
                 'tense': 6158,
                 'capitalism': 6617,
                 'simplified': 8530,
                 'twins': 7153,
                 'withdrew': 8845,
                 'folks': 5259,
                 'soviet': 805,
                 'revival': 9962,
                 'feasible': 6010,
                 'spring': 834,
                 'ash': 7586,
                 'translated': 5934,
                 'meant': 1075,
                 '2:35': 9029,
                 'batting': 6170,
                 'males': 5063,
                 'needless': 7466,
                 'juniors': 3480,
                 'barton': 4120,
                 'radius': 9027,
                 'earthquakes': 8846,
                 'extra': 2248,
                 'lower': 855,
                 'shaking': 4613,
                 'remember': 750,
                 'cook': 2368,
                 'fabrics': 3650,
                 'adding': 4532,
                 'extruded': 9144,
                 'claims': 1469,
                 'patchen': 3930,
                 'garson': 8803,
                 'deduction': 7353,
                 'minimal': 3855,
                 'sessions': 3931,
                 'decision': 889,
                 'unexpected': 4298,
                 'differently': 5926,
                 'unknown': 2359,
                 'derive': 6831,
                 'loading': 7587,
                 'yourself': 1636,
                 'lalaurie': 5575,
                 'adequately': 5765,
                 'embrace': 6842,
                 'golf': 3203,
                 'plow': 7243,
                 'acre': 7987,
                 'collector': 9261,
                 'policy': 432,
                 'perhaps': 317,
                 'repel': 9395,
                 'hardy': 2626,
                 'dusk': 9030,
                 'boom': 9617,
                 'access': 4205,
                 'convinced': 2222,
                 'minor': 1923,
                 'decreased': 9229,
                 'stretching': 5637,
                 'cerebral': 9618,
                 'defendants': 8093,
                 'clarity': 3719,
                 'presumably': 2766,
                 'participating': 6011,
                 'bluff': 9396,
                 'cosmic': 5316,
                 'ma': 5156,
                 'plug': 4349,
                 'via': 2322,
                 'parameters': 9824,
                 'itself': 318,
                 'disturbing': 5835,
                 'essex': 7258,
                 'living': 512,
                 'religions': 5210,
                 'across': 342,
                 'churchyard': 9825,
                 'sands': 8184,
                 'optimal': 3716,
                 'believed': 1408,
                 'stravinsky': 9786,
                 'china': 1590,
                 'core': 2568,
                 'might': 150,
                 'eighty': 7713,
                 'couples': 6992,
                 'clocks': 9826,
                 'lone': 9651,
                 'applause': 6381,
                 'parents': 1183,
                 'softly': 3501,
                 "one's": 1704,
                 'occasionally': 2297,
                 'keeps': 4760,
                 'election': 1415,
                 'challenging': 7187,
                 'luxury': 4761,
                 'spare': 4419,
                 'circuit': 4299,
                 'regional': 2645,
                 'athlete': 8675,
                 'breathed': 8512,
                 'installations': 5688,
                 'simply': 591,
                 'bombs': 3141,
                 'specially': 9031,
                 'ludie': 6932,
                 'cleveland': 5577,
                 'rivers': 5498,
                 'undesirable': 8411,
                 "it's": 320,
                 'may': 82,
                 'naturally': 1550,
                 'achieve': 2197,
                 'runs': 2052,
                 'proclamation': 6309,
                 'channels': 4379,
                 'run': 454,
                 'grimly': 7714,
                 'jenny': 9418,
                 'gossip': 6728,
                 'iliad': 6554,
                 'stunned': 9619,
                 'steps': 892,
                 "hasn't": 4920,
                 'stocks': 5380,
                 'cohesive': 7916,
                 'paused': 3794,
                 'receive': 1431,
                 'distinguish': 5157,
                 'told': 222,
                 'social-class': 9827,
                 'roare




<b>limit_output extension: Maximum message size of 10000 exceeded with 29879 characters</b>



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
    [[   2 3084    5    2 2237 1330 4237 2431    3    6   19   26 1067   69
         8 2083    6    3    1    3  266   65  342    2    1    3    2  315
         1    9   87  216 3291   69 1568    4    0    0    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0]
     [  45   12    8  512 8109    6   60 3230   39    2    1    1    3    2
       849    1    3    1    3   10 9813    2    1 3458    9   43    1    1
         3    6    2 1050  385   73 4569    3    9    2    1    1 3213    3
        12   10    2  855 5425   12    8 8888  121    1    4]
     [  33   64   26   12  443    7 7062    9    8 3327    3    1 2761    3
         2  463  571    2    1    1 1644   12    1    4    0    0    0    0
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

    /home/karen/.local/lib/python3.5/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    Using TensorFlow backend.


    WARNING:tensorflow:From /home/karen/.local/lib/python3.5/site-packages/keras/backend/tensorflow_backend.py:1247: calling reduce_sum (from tensorflow.python.ops.math_ops) with keep_dims is deprecated and will be removed in a future version.
    Instructions for updating:
    keep_dims is deprecated, use keepdims instead


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

    WARNING:tensorflow:From /home/karen/.local/lib/python3.5/site-packages/keras/backend/tensorflow_backend.py:1349: calling reduce_mean (from tensorflow.python.ops.math_ops) with keep_dims is deprecated and will be removed in a future version.
    Instructions for updating:
    keep_dims is deprecated, use keepdims instead
    Epoch 1/5
     228/1343 [====>.........................] - ETA: 22s - loss: 0.90


<b>limit_output extension: Maximum message size of 10000 exceeded with 10131 characters</b>


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
    
    Current values:
    NotebookApp.iopub_data_rate_limit=1000000.0 (bytes/sec)
    NotebookApp.rate_limit_window=3.0 (secs)
    


    13120/14335 [==========================>...] - ETA: 


<b>limit_output extension: Maximum message size of 10000 exceeded with 10032 characters</b>


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
     183/1343 [===>..........................] - ETA: 30s - loss: 0.76


<b>limit_output extension: Maximum message size of 10000 exceeded with 10097 characters</b>



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
    
    Current values:
    NotebookApp.iopub_data_rate_limit=1000000.0 (bytes/sec)
    NotebookApp.rate_limit_window=3.0 (secs)
    


     9600/14335 [===================>..........] - ETA: 


<b>limit_output extension: Maximum message size of 10000 exceeded with 10091 characters</b>


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
      72/1343 [>.............................] - ETA: 2:39 - loss: 1.27


<b>limit_output extension: Maximum message size of 10000 exceeded with 10099 characters</b>



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

    (('open', 'market', 'policy'), ('and', 'you', 'think', 'you', 'have', 'language', 'problems', '.'), ('mae', 'entered', 'the', 'room', 'from', 'the', 'hallway', 'to', 'the', 'kitchen', '.'), ('this', 'will', 'permit', 'you', 'to', 'get', 'a', 'rough', 'estimate', 'of', 'how', 'much', 'the', 'materials', 'for', 'the', 'shell', 'will', 'cost', '.'), ('the', 'multifigure', '``', 'traveling', 'carnival', "''", ',', 'in', 'which', 'action', 'is', 'vivified', 'by', 'lighting', ';', ';'), ('they', 'are', 'in', 'general', 'those', 'fears', 'that', 'once', 'seemed', 'to', 'have', 'been', 'amenable', 'to', 'prayer', 'or', 'ritual', '.'), ('yet', 'they', 'are', 'written', ';', ';'), ('the', 'plantation', 'was', 'sold', 'in', 'january', ',', '1845', ',', 'and', 'palfrey', 'thought', 'the', 'new', 'owner', 'ought', 'to', 'pay', 'his', 'people', 'two', "months'", 'wages', '.'), ('this', 'is', 'the', 'end', 'of', 'the', 'line', "''", '.'), ('the', 'rules', 'and', 'policies', 'to', 'be', 'applied', 'in', 'this', 'process', 'of', 'course', 'must', 'be', 'based', 'on', 'objectives', 'which', 'represent', 'what', 'is', 'to', 'be', 'desired', 'if', 'radio', 'service', 'is', 'to', 'be', 'of', 'maximum', 'use', 'to', 'the', 'nation', '.'), ('a', 'mirror', 'is', 'mounted', 'on', 'each', 'accelerometer', 'so', 'that', 'the', 'plane', 'of', 'the', 'mirror', 'is', 'perpendicular', 'to', 'the', 'sensitive', 'axis', 'of', 'the', 'unit', '.'), ('she', 'stayed', 'away', 'for', 'ten', 'days', '.'), ('``', "i'd", 'druther', 'stay', 'here', 'and', 'watch', 'the', 'girls', "''", ',', 'charles', 'grinned', '.'), ('said', 'trenchard', ',', 'otherwise', 'hawk', '.'), ('wexler', 'has', 'denied', 'repeatedly', 'that', 'coercion', 'was', 'used', 'in', 'questioning', '.'), ('referring', 'further', 'to', 'the', "foundation's", 'officers', ',', 'dr.', 'james', 'f.', 'mathias', ',', 'for', 'eleven', 'years', 'our', 'discerning', 'colleague', 'as', 'associate', 'secretary', ',', 'was', 'promoted', 'to', 'be', 'secretary', '.'), ('``', 'jackson', 'recruited', 'his', 'critters', ',', 'and', 'him', 'and', 'me', 'fixed', 'up', 'his', 'wagon', 'while', 'we', 'was', 'waiting', 'for', 'you', 'to', 'catch', 'up', '.'), ('he', 'might', 'have', 'been', 'anywhere', 'or', 'nowhere', '.'), ('in', 'a', 'fraction', 'of', 'a', 'second', 'the', 'pickup', 'truck', 'hurtled', 'by', 'on', 'the', 'other', 'side', '.'), ('and', 'if', 'we', 'do', 'not', 'aspire', 'to', 'too', 'much', ',', 'it', 'is', 'also', 'within', 'our', 'capacity', '.'), ('``', 'there', 'was', 'only', 'one', 'power', 'control', '--', 'a', 'valve', 'to', 'adjust', 'the', 'fuel', 'flow', '.'), ('both', 'cars', 'were', 'slightly', 'damaged', '.'), ('in', 'the', 'field', 'of', 'political', 'values', ',', 'it', 'is', 'certainly', 'true', 'that', 'students', 'are', 'not', 'radical', ',', 'not', 'rebels', 'against', 'their', 'parents', 'or', 'their', 'peers', '.'), ('``', 'i', 'will', 'go', "''", '.'), ('an', 'interregnum', 'ensues', 'in', 'which', 'not', 'men', 'but', 'ideas', 'compete', 'for', 'existence', '.'), ('do', 'you', 'protect', 'your', 'holiday', 'privileges', 'with', 'an', 'attendance', 'requirement', 'both', 'before', 'and', 'after', 'the', 'holiday', '?', '?'), ('being', 'an', 'intelligent', 'man', ',', 'john', 'must', 'have', 'guessed', 'what', 'everyone', 'thought', 'about', 'edythe', ',', 'but', 'he', 'never', 'let', 'on', 'by', 'so', 'much', 'as', 'a', 'brave', 'smile', '.'), ('he', 'sat', 'down', 'next', 'to', 'a', 'heavily-upholstered', 'blonde', ',', 'but', 'she', 'was', 'cleaned', 'out', 'in', 'twenty', 'minutes', '.'), ('for', 'them', ',', 'in', 'the', 'grim', 'words', 'of', 'a', 'once-popular', 'song', ',', 'love', 'and', 'marriage', 'go', 'together', 'like', 'a', 'horse', 'and', 'carriage', '.'), ('the', 'voice', 'had', 'music', 'in', 'it', '.'), ('one', 'night', ',', 'mama', 'came', 'home', 'practically', 'in', 'a', 'state', 'of', 'shock', '.'), ('welch', 'was', 'wild', 'with', 'delight', '.'), ('to', 'further', 'increase', 'back', 'flexibility', ',', 'work', 'on', 'the', 'back', 'circle', '.'), ('in', '1913', 'an', 'abortive', 'provision', 'was', 'made', 'for', 'the', 'stay', 'of', 'federal', 'injunction', 'proceedings', 'upon', 'institution', 'of', 'state', 'court', 'test', 'cases', '.'), ('and', 'you', 'also', 'got', 'this', 'little', 'spark', 'in', 'your', 'bird-brain', 'that', 'tells', 'you', 'to', 'turn', 'around', 'before', 'you', 'drown', 'yourself', '.'), ('i', "didn't", 'hurry', 'though', 'it', 'was', 'cold', 'and', 'the', 'pedersen', 'kid', 'was', 'in', 'the', 'kitchen', '.'), ('otherwise', ',', 'you', 'may', 'be', 'saddled', 'with', 'a', 'good-size', 'milk', 'bill', 'by', 'milk', 'drinkers', '.'), ('in', 'a', 'letter', 'to', 'the', 'american', 'friends', 'service', ',', 'dr.', 'schweitzer', 'wrote', ':'), ('apparently', 'sensing', 'this', ',', 'and', 'realizing', 'that', 'it', 'gave', 'him', 'an', 'advantage', ',', 'jess', 'became', 'bold', '.'), ('pilot', 'plant', 'operations'), ('in', 'his', 'recognition', 'of', 'his', 'impersonal', 'self', 'the', 'dancer', 'moves', ',', 'and', 'this', 'self', ',', 'in', 'the', '``', 'first', 'revealed', 'stroke', 'of', 'its', 'existence', "''", ',', 'states', 'the', 'theme', 'from', 'which', 'all', 'else', 'must', 'follow', '.'), ('``', "it's", 'a', 'sublease', '.'), ('is', 'this', 'site', 'available', '?', '?'), ('the', 'paper', 'affords', 'excellent', 'practice', 'for', 'students', 'interested', 'in', 'the', 'field', 'of', 'journalism', '.'), ('i', 'say', 'the', 'late', 'seventeenth', 'century', 'because', 'racine', '(', 'whom', 'lessing', 'did', 'not', 'really', 'know', ')', 'stands', 'on', 'the', 'far', 'side', 'of', 'the', 'chasm', '.'), ('fosterite', 'bishops', ',', 'after', 'secret', 'conclave', ',', 'announced', 'the', "church's", 'second', 'major', 'miracle', ':', 'supreme', 'bishop', 'digby', 'had', 'been', 'translated', 'bodily', 'to', 'heaven', 'and', 'spot-promoted', 'to', 'archangel', ',', 'ranking', 'with-but-after', 'archangel', 'foster', '.'), ('he', 'identified', 'the', 'man', 'as', 'lewis', 'martin', 'parker', ',', '59', ',', 'a', 'farmer', 'of', 'hartselle', ',', 'ala.', '.'), ('it', 'reappears', ',', 'in', 'whole', 'or', 'part', ',', 'whenever', 'a', 'new', 'crisis', 'exposes', 'the', 'reality', ':', 'in', 'cuba', 'last', 'spring', '(', 'with', 'which', 'the', 'dominican', 'events', 'of', 'last', 'month', 'should', 'be', 'paired', ')', ';', ';'), ('extend', 'your', 'feet', 'forward', 'and', 'backward', 'until', 'you', 'are', 'in', 'a', 'deep', 'leg', 'split', '.'), ('a', 'certain', 'skepticism', 'about', 'the', 'coming', 'of', 'americans', 'is', 'to', 'be', 'expected', 'in', 'many', 'quarters', '.'), ('the', 'crux', 'of', 'ecumenical', 'advance', 'is', 'an', 'even', 'more', 'personalized', 'matter', 'than', 'the', 'relation', 'between', 'congregations', 'in', 'the', 'same', 'community', '.'), ('issue', 'no.', '5', '.'), ('but', 'this', 'was', 'only', 'the', 'middle', 'of', 'july', '.'), ('``', 'gyp', 'carmer', "couldn't", 'have', 'known', 'about', "colcord's", 'money', 'unless', 'he', 'was', 'told', '--', 'and', 'who', 'else', 'would', 'have', 'told', 'him', "''", '?', '?'), ('perhaps', 'he', 'had', 'better', 'have', 'someone', 'help', 'him', 'put', 'up', 'the', 'pegboard', 'and', 'build', 'the', 'workbench', '--', 'someone', 'who', 'knew', 'what', 'he', 'was', 'about', '.'), ('last', 'year', ',', 'we', 'probably', 'would', 'have', 'given', 'him', '$700', 'for', 'a', 'comparable', 'machine', "''", '.'), ('however', ',', 'she', 'really', 'does', 'not', 'know', 'how', 'to', 'match', 'the', 'quantity', 'of', 'dollars', 'given', 'away', 'by', 'a', 'quality', 'of', 'leadership', 'that', 'is', 'basically', 'needed', '.'), ('despite', 'extensive', 'attempts', 'to', 'obtain', 'highly', 'pure', 'reagents', ',', 'serious', 'difficulty', 'was', 'experienced', 'in', 'obtaining', 'reproducible', 'rates', 'of', 'reaction', '.'), ('at', 'fifteen', 'he', "didn't", 'care', 'that', 'he', 'had', 'no', 'mother', ',', 'that', 'he', "couldn't", 'remember', 'her', 'face', 'or', 'her', 'touch', ';', ';'), ('even', 'after', 'the', 'incident', 'between', 'bang-jensen', 'and', 'shann', 'in', 'the', "delegates'", 'lounge', 'and', 'this', 'was', 'not', 'the', 'way', 'the', 'chicago', 'tribune', 'presented', 'it', "''", '.'), ('in', 'order', 'to', 'attract', 'new', 'industries', ',', '15', 'states', 'or', 'more', 'are', 'issuing', 'tax', 'free', 'bonds', 'to', 'build', 'government', 'owned', 'plants', 'which', 'are', 'leased', 'to', 'private', 'enterprise', '.'), ('(', 'each', "state's", 'unadjusted', 'allotment', 'for', 'any', 'fiscal', 'year', ',', 'which', 'exceeds', 'its', 'minimum', 'allotment', 'described', 'in', 'item', '13', 'below', 'by', 'a', 'percentage', 'greater', 'than', 'one', 'and', 'one-half', 'times', 'the', 'percentage', 'by', 'which', 'the', 'sum', 'being', 'allotted', 'exceeds', '$23,000,000', ',', 'must', 'be', 'reduced', 'by', 'the', 'amount', 'of', 'the', 'excess', '.'), ('but', 'a', 'historian', 'might', 'put', 'his', 'finger', 'on', 'a', 'specific', 'man', 'and', 'date', ',', 'and', 'hold', 'out', 'the', 'hope', 'that', 'the', 'troubles', 'will', 'sometime', 'pass', 'away', '.'), ('very', 'small', 'concentrations', 'of', 'these', 'hydrides', 'should', 'be', 'detectable', ';', ';'), ('do', 'something', "''", '!', '!'), ('of', 'course', 'the', 'principal', 'factor', 'in', 'the', 'whole', 'experience', 'was', 'the', 'kind', 'of', 'education', 'he', 'received', '.'), ('this', 'will', 'help', 'him', 'to', 'get', 'out', 'of', 'his', 'little', 'tackle', 'shop', '.'), ('he', 'came', 'to', 'the', 'edge', 'of', 'the', 'veranda', ',', 'peered', 'down', 'at', 'them', 'with', 'his', 'hand', 'on', 'his', 'gun', '.'), ("mississippi's", 'relations', 'with', 'the', 'national', 'democratic', 'party', 'will', 'be', 'at', 'a', 'crossroads', 'during', '1961', ',', 'with', 'the', 'first', 'democratic', 'president', 'in', 'eight', 'years', 'in'


<b>limit_output extension: Maximum message size of 10000 exceeded with 2435477 characters</b>


```

```

```

```

```

```

```

```

```

```

```

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

