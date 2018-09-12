
## Intro and text classification

__Rule-based methods__
- Regular expressions
- Semantic slot filling: CFG
    - Context-free grammars
    
![](../../images/1.png)


![](../../images/2.png)

![](../../images/3.png)

    


__Probabilistic modeling and machine learning__
- Likelihood maximization
- Linear classifiers

        Perform good enough in many tasks
            - eg. sequence labeling
        Allow us not to be blinded with the hype
            - eg. word2vec / distributional semantics
        Help to further improve DL models
            - eg. word alignment prior in machine translation


__Deep Learning__
- RNN

![](../../images/4.png)

- CNN






















## Simple recap of the application of NLP

![](../../images/5.png)

![](../../images/6.png)

![](../../images/7.png)

![](../../images/8.png)

![](../../images/9.png)

![](../../images/10.png)

![](../../images/11.png)

![](../../images/12.png)


- Libraries

![](../../images/13.png)

![](../../images/14.png)

![](../../images/15.png)

![](../../images/16.png)

![](../../images/17.png)



## Implementation: Text preprocessing

### Additional notes: __all the taggers in nltk__


#### pos_tag


```python
import nltk
# pos_tag (pos_tag load the Standard treebank POS tagger)
text = nltk.word_tokenize("And now for something completely different")
nltk.pos_tag(text)
```




    [('And', 'CC'),
     ('now', 'RB'),
     ('for', 'IN'),
     ('something', 'NN'),
     ('completely', 'RB'),
     ('different', 'JJ')]



![](../../images/18.png)




```python
 nltk.corpus.brown.tagged_words()
```




    [('The', 'AT'), ('Fulton', 'NP-TL'), ...]



#### Automatic Tagging


```python
# 因为tag要根据词的context，所以tag是以sentense为单位的，而不是word为单位，因为如果以词为单位，一个句子的结尾词会影响到下个句子开头词的tag，
# 这样是不合理的，以句子为单位可以避免这样的错误，让context的影响不会越过sentense

from nltk.corpus import brown
brown_tagged_sents = brown.tagged_sents(categories='news')
brown_tagged_sents
```




    [[('The', 'AT'), ('Fulton', 'NP-TL'), ('County', 'NN-TL'), ('Grand', 'JJ-TL'), ('Jury', 'NN-TL'), ('said', 'VBD'), ('Friday', 'NR'), ('an', 'AT'), ('investigation', 'NN'), ('of', 'IN'), ("Atlanta's", 'NP$'), ('recent', 'JJ'), ('primary', 'NN'), ('election', 'NN'), ('produced', 'VBD'), ('``', '``'), ('no', 'AT'), ('evidence', 'NN'), ("''", "''"), ('that', 'CS'), ('any', 'DTI'), ('irregularities', 'NNS'), ('took', 'VBD'), ('place', 'NN'), ('.', '.')], [('The', 'AT'), ('jury', 'NN'), ('further', 'RBR'), ('said', 'VBD'), ('in', 'IN'), ('term-end', 'NN'), ('presentments', 'NNS'), ('that', 'CS'), ('the', 'AT'), ('City', 'NN-TL'), ('Executive', 'JJ-TL'), ('Committee', 'NN-TL'), (',', ','), ('which', 'WDT'), ('had', 'HVD'), ('over-all', 'JJ'), ('charge', 'NN'), ('of', 'IN'), ('the', 'AT'), ('election', 'NN'), (',', ','), ('``', '``'), ('deserves', 'VBZ'), ('the', 'AT'), ('praise', 'NN'), ('and', 'CC'), ('thanks', 'NNS'), ('of', 'IN'), ('the', 'AT'), ('City', 'NN-TL'), ('of', 'IN-TL'), ('Atlanta', 'NP-TL'), ("''", "''"), ('for', 'IN'), ('the', 'AT'), ('manner', 'NN'), ('in', 'IN'), ('which', 'WDT'), ('the', 'AT'), ('election', 'NN'), ('was', 'BEDZ'), ('conducted', 'VBN'), ('.', '.')], ...]



#### The Regular Expression Tagger


```python
patterns = [
(r'.*ing$', 'VBG'), # gerunds
(r'.*ed$', 'VBD'), # simple past
(r'.*es$', 'VBZ'), # 3rd singular present
(r'.*ould$', 'MD'), # modals
(r'.*\'s$', 'NN'), # possessive nouns
(r'.*s$', 'NNS'), # plural nouns
(r'^-?[0-9]+(.[0-9]+)?$', 'CD'), # cardinal numbers
(r'.*ly$', 'RB'), # adv
(r'.*', 'NN')] # nouns (default)
regexp_tagger = nltk.RegexpTagger(patterns)
regexp_tagger.tag(['And', 'now', 'for', 'something', 'completely', 'different'])
regexp_tagger.tag(brown.words(categories='news'))
```




    [('The', 'NN'),
     ('Fulton', 'NN'),
     ('County', 'NN'),
     ('Grand', 'NN'),
     ('Jury', 'NN'),
     ('said', 'NN'),
     ('Friday', 'NN'),
     ('an', 'NN'),
     ('investigation', 'NN'),
     ('of', 'NN'),
     ("Atlanta's", 'NN'),
     ('recent', 'NN'),
     ('primary', 'NN'),
     ('election', 'NN'),
     ('produced', 'VBD'),
     ('``', 'NN'),
     ('no', 'NN'),
     ('evidence', 'NN'),
     ("''", 'NN'),
     ('that', 'NN'),
     ('any', 'NN'),
     ('irregularities', 'VBZ'),
     ('took', 'NN'),
     ('place', 'NN'),
     ('.', 'NN'),
     ('The', 'NN'),
     ('jury', 'NN'),
     ('further', 'NN'),
     ('said', 'NN'),
     ('in', 'NN'),
     ('term-end', 'NN'),
     ('presentments', 'NNS'),
     ('that', 'NN'),
     ('the', 'NN'),
     ('City', 'NN'),
     ('Executive', 'NN'),
     ('Committee', 'NN'),
     (',', 'NN'),
     ('which', 'NN'),
     ('had', 'NN'),
     ('over-all', 'NN'),
     ('charge', 'NN'),
     ('of', 'NN'),
     ('the', 'NN'),
     ('election', 'NN'),
     (',', 'NN'),
     ('``', 'NN'),
     ('deserves', 'VBZ'),
     ('the', 'NN'),
     ('praise', 'NN'),
     ('and', 'NN'),
     ('thanks', 'NNS'),
     ('of', 'NN'),
     ('the', 'NN'),
     ('City', 'NN'),
     ('of', 'NN'),
     ('Atlanta', 'NN'),
     ("''", 'NN'),
     ('for', 'NN'),
     ('the', 'NN'),
     ('manner', 'NN'),
     ('in', 'NN'),
     ('which', 'NN'),
     ('the', 'NN'),
     ('election', 'NN'),
     ('was', 'NNS'),
     ('conducted', 'VBD'),
     ('.', 'NN'),
     ('The', 'NN'),
     ('September-October', 'NN'),
     ('term', 'NN'),
     ('jury', 'NN'),
     ('had', 'NN'),
     ('been', 'NN'),
     ('charged', 'VBD'),
     ('by', 'NN'),
     ('Fulton', 'NN'),
     ('Superior', 'NN'),
     ('Court', 'NN'),
     ('Judge', 'NN'),
     ('Durwood', 'NN'),
     ('Pye', 'NN'),
     ('to', 'NN'),
     ('investigate', 'NN'),
     ('reports', 'NNS'),
     ('of', 'NN'),
     ('possible', 'NN'),
     ('``', 'NN'),
     ('irregularities', 'VBZ'),
     ("''", 'NN'),
     ('in', 'NN'),
     ('the', 'NN'),
     ('hard-fought', 'NN'),
     ('primary', 'NN'),
     ('which', 'NN'),
     ('was', 'NNS'),
     ('won', 'NN'),
     ('by', 'NN'),
     ('Mayor-nominate', 'NN'),
     ('Ivan', 'NN'),
     ('Allen', 'NN'),
     ('Jr.', 'NN'),
     ('.', 'NN'),
     ('``', 'NN'),
     ('Only', 'RB'),
     ('a', 'NN'),
     ('relative', 'NN'),
     ('handful', 'NN'),
     ('of', 'NN'),
     ('such', 'NN'),
     ('reports', 'NNS'),
     ('was', 'NNS'),
     ('received', 'VBD'),
     ("''", 'NN'),
     (',', 'NN'),
     ('the', 'NN'),
     ('jury', 'NN'),
     ('said', 'NN'),
     (',', 'NN'),
     ('``', 'NN'),
     ('considering', 'VBG'),
     ('the', 'NN'),
     ('widespread', 'NN'),
     ('interest', 'NN'),
     ('in', 'NN'),
     ('the', 'NN'),
     ('election', 'NN'),
     (',', 'NN'),
     ('the', 'NN'),
     ('number', 'NN'),
     ('of', 'NN'),
     ('voters', 'NNS'),
     ('and', 'NN'),
     ('the', 'NN'),
     ('size', 'NN'),
     ('of', 'NN'),
     ('this', 'NNS'),
     ('city', 'NN'),
     ("''", 'NN'),
     ('.', 'NN'),
     ('The', 'NN'),
     ('jury', 'NN'),
     ('said', 'NN'),
     ('it', 'NN'),
     ('did', 'NN'),
     ('find', 'NN'),
     ('that', 'NN'),
     ('many', 'NN'),
     ('of', 'NN'),
     ("Georgia's", 'NN'),
     ('registration', 'NN'),
     ('and', 'NN'),
     ('election', 'NN'),
     ('laws', 'NNS'),
     ('``', 'NN'),
     ('are', 'NN'),
     ('outmoded', 'VBD'),
     ('or', 'NN'),
     ('inadequate', 'NN'),
     ('and', 'NN'),
     ('often', 'NN'),
     ('ambiguous', 'NNS'),
     ("''", 'NN'),
     ('.', 'NN'),
     ('It', 'NN'),
     ('recommended', 'VBD'),
     ('that', 'NN'),
     ('Fulton', 'NN'),
     ('legislators', 'NNS'),
     ('act', 'NN'),
     ('``', 'NN'),
     ('to', 'NN'),
     ('have', 'NN'),
     ('these', 'NN'),
     ('laws', 'NNS'),
     ('studied', 'VBD'),
     ('and', 'NN'),
     ('revised', 'VBD'),
     ('to', 'NN'),
     ('the', 'NN'),
     ('end', 'NN'),
     ('of', 'NN'),
     ('modernizing', 'VBG'),
     ('and', 'NN'),
     ('improving', 'VBG'),
     ('them', 'NN'),
     ("''", 'NN'),
     ('.', 'NN'),
     ('The', 'NN'),
     ('grand', 'NN'),
     ('jury', 'NN'),
     ('commented', 'VBD'),
     ('on', 'NN'),
     ('a', 'NN'),
     ('number', 'NN'),
     ('of', 'NN'),
     ('other', 'NN'),
     ('topics', 'NNS'),
     (',', 'NN'),
     ('among', 'NN'),
     ('them', 'NN'),
     ('the', 'NN'),
     ('Atlanta', 'NN'),
     ('and', 'NN'),
     ('Fulton', 'NN'),
     ('County', 'NN'),
     ('purchasing', 'VBG'),
     ('departments', 'NNS'),
     ('which', 'NN'),
     ('it', 'NN'),
     ('said', 'NN'),
     ('``', 'NN'),
     ('are', 'NN'),
     ('well', 'NN'),
     ('operated', 'VBD'),
     ('and', 'NN'),
     ('follow', 'NN'),
     ('generally', 'RB'),
     ('accepted', 'VBD'),
     ('practices', 'VBZ'),
     ('which', 'NN'),
     ('inure', 'NN'),
     ('to', 'NN'),
     ('the', 'NN'),
     ('best', 'NN'),
     ('interest', 'NN'),
     ('of', 'NN'),
     ('both', 'NN'),
     ('governments', 'NNS'),
     ("''", 'NN'),
     ('.', 'NN'),
     ('Merger', 'NN'),
     ('proposed', 'VBD'),
     ('However', 'NN'),
     (',', 'NN'),
     ('the', 'NN'),
     ('jury', 'NN'),
     ('said', 'NN'),
     ('it', 'NN'),
     ('believes', 'VBZ'),
     ('``', 'NN'),
     ('these', 'NN'),
     ('two', 'NN'),
     ('offices', 'VBZ'),
     ('should', 'MD'),
     ('be', 'NN'),
     ('combined', 'VBD'),
     ('to', 'NN'),
     ('achieve', 'NN'),
     ('greater', 'NN'),
     ('efficiency', 'NN'),
     ('and', 'NN'),
     ('reduce', 'NN'),
     ('the', 'NN'),
     ('cost', 'NN'),
     ('of', 'NN'),
     ('administration', 'NN'),
     ("''", 'NN'),
     ('.', 'NN'),
     ('The', 'NN'),
     ('City', 'NN'),
     ('Purchasing', 'VBG'),
     ('Department', 'NN'),
     (',', 'NN'),
     ('the', 'NN'),
     ('jury', 'NN'),
     ('said', 'NN'),
     (',', 'NN'),
     ('``', 'NN'),
     ('is', 'NNS'),
     ('lacking', 'VBG'),
     ('in', 'NN'),
     ('experienced', 'VBD'),
     ('clerical', 'NN'),
     ('personnel', 'NN'),
     ('as', 'NNS'),
     ('a', 'NN'),
     ('result', 'NN'),
     ('of', 'NN'),
     ('city', 'NN'),
     ('personnel', 'NN'),
     ('policies', 'VBZ'),
     ("''", 'NN'),
     ('.', 'NN'),
     ('It', 'NN'),
     ('urged', 'VBD'),
     ('that', 'NN'),
     ('the', 'NN'),
     ('city', 'NN'),
     ('``', 'NN'),
     ('take', 'NN'),
     ('steps', 'NNS'),
     ('to', 'NN'),
     ('remedy', 'NN'),
     ("''", 'NN'),
     ('this', 'NNS'),
     ('problem', 'NN'),
     ('.', 'NN'),
     ('Implementation', 'NN'),
     ('of', 'NN'),
     ("Georgia's", 'NN'),
     ('automobile', 'NN'),
     ('title', 'NN'),
     ('law', 'NN'),
     ('was', 'NNS'),
     ('also', 'NN'),
     ('recommended', 'VBD'),
     ('by', 'NN'),
     ('the', 'NN'),
     ('outgoing', 'VBG'),
     ('jury', 'NN'),
     ('.', 'NN'),
     ('It', 'NN'),
     ('urged', 'VBD'),
     ('that', 'NN'),
     ('the', 'NN'),
     ('next', 'NN'),
     ('Legislature', 'NN'),
     ('``', 'NN'),
     ('provide', 'NN'),
     ('enabling', 'VBG'),
     ('funds', 'NNS'),
     ('and', 'NN'),
     ('re-set', 'NN'),
     ('the', 'NN'),
     ('effective', 'NN'),
     ('date', 'NN'),
     ('so', 'NN'),
     ('that', 'NN'),
     ('an', 'NN'),
     ('orderly', 'RB'),
     ('implementation', 'NN'),
     ('of', 'NN'),
     ('the', 'NN'),
     ('law', 'NN'),
     ('may', 'NN'),
     ('be', 'NN'),
     ('effected', 'VBD'),
     ("''", 'NN'),
     ('.', 'NN'),
     ('The', 'NN'),
     ('grand', 'NN'),
     ('jury', 'NN'),
     ('took', 'NN'),
     ('a', 'NN'),
     ('swipe', 'NN'),
     ('at', 'NN'),
     ('the', 'NN'),
     ('State', 'NN'),
     ('Welfare', 'NN'),
     ("Department's", 'NN'),
     ('handling', 'VBG'),
     ('of', 'NN'),
     ('federal', 'NN'),
     ('funds', 'NNS'),
     ('granted', 'VBD'),
     ('for', 'NN'),
     ('child', 'NN'),
     ('welfare', 'NN'),
     ('services', 'VBZ'),
     ('in', 'NN'),
     ('foster', 'NN'),
     ('homes', 'VBZ'),
     ('.', 'NN'),
     ('``', 'NN'),
     ('This', 'NNS'),
     ('is', 'NNS'),
     ('one', 'NN'),
     ('of', 'NN'),
     ('the', 'NN'),
     ('major', 'NN'),
     ('items', 'NNS'),
     ('in', 'NN'),
     ('the', 'NN'),
     ('Fulton', 'NN'),
     ('County', 'NN'),
     ('general', 'NN'),
     ('assistance', 'NN'),
     ('program', 'NN'),
     ("''", 'NN'),
     (',', 'NN'),
     ('the', 'NN'),
     ('jury', 'NN'),
     ('said', 'NN'),
     (',', 'NN'),
     ('but', 'NN'),
     ('the', 'NN'),
     ('State', 'NN'),
     ('Welfare', 'NN'),
     ('Department', 'NN'),
     ('``', 'NN'),
     ('has', 'NNS'),
     ('seen', 'NN'),
     ('fit', 'NN'),
     ('to', 'NN'),
     ('distribute', 'NN'),
     ('these', 'NN'),
     ('funds', 'NNS'),
     ('through', 'NN'),
     ('the', 'NN'),
     ('welfare', 'NN'),
     ('departments', 'NNS'),
     ('of', 'NN'),
     ('all', 'NN'),
     ('the', 'NN'),
     ('counties', 'VBZ'),
     ('in', 'NN'),
     ('the', 'NN'),
     ('state', 'NN'),
     ('with', 'NN'),
     ('the', 'NN'),
     ('exception', 'NN'),
     ('of', 'NN'),
     ('Fulton', 'NN'),
     ('County', 'NN'),
     (',', 'NN'),
     ('which', 'NN'),
     ('receives', 'VBZ'),
     ('none', 'NN'),
     ('of', 'NN'),
     ('this', 'NNS'),
     ('money', 'NN'),
     ('.', 'NN'),
     ('The', 'NN'),
     ('jurors', 'NNS'),
     ('said', 'NN'),
     ('they', 'NN'),
     ('realize', 'NN'),
     ('``', 'NN'),
     ('a', 'NN'),
     ('proportionate', 'NN'),
     ('distribution', 'NN'),
     ('of', 'NN'),
     ('these', 'NN'),
     ('funds', 'NNS'),
     ('might', 'NN'),
     ('disable', 'NN'),
     ('this', 'NNS'),
     ('program', 'NN'),
     ('in', 'NN'),
     ('our', 'NN'),
     ('less', 'NNS'),
     ('populous', 'NNS'),
     ('counties', 'VBZ'),
     ("''", 'NN'),
     ('.', 'NN'),
     ('Nevertheless', 'NNS'),
     (',', 'NN'),
     ('``', 'NN'),
     ('we', 'NN'),
     ('feel', 'NN'),
     ('that', 'NN'),
     ('in', 'NN'),
     ('the', 'NN'),
     ('future', 'NN'),
     ('Fulton', 'NN'),
     ('County', 'NN'),
     ('should', 'MD'),
     ('receive', 'NN'),
     ('some', 'NN'),
     ('portion', 'NN'),
     ('of', 'NN'),
     ('these', 'NN'),
     ('available', 'NN'),
     ('funds', 'NNS'),
     ("''", 'NN'),
     (',', 'NN'),
     ('the', 'NN'),
     ('jurors', 'NNS'),
     ('said', 'NN'),
     ('.', 'NN'),
     ('``', 'NN'),
     ('Failure', 'NN'),
     ('to', 'NN'),
     ('do', 'NN'),
     ('this', 'NNS'),
     ('will', 'NN'),
     ('continue', 'NN'),
     ('to', 'NN'),
     ('place', 'NN'),
     ('a', 'NN'),
     ('disproportionate', 'NN'),
     ('burden', 'NN'),
     ("''", 'NN'),
     ('on', 'NN'),
     ('Fulton', 'NN'),
     ('taxpayers', 'NNS'),
     ('.', 'NN'),
     ('The', 'NN'),
     ('jury', 'NN'),
     ('also', 'NN'),
     ('commented', 'VBD'),
     ('on', 'NN'),
     ('the', 'NN'),
     ('Fulton', 'NN'),
     ("ordinary's", 'NN'),
     ('court', 'NN'),
     ('which', 'NN'),
     ('has', 'NNS'),
     ('been', 'NN'),
     ('under', 'NN'),
     ('fire', 'NN'),
     ('for', 'NN'),
     ('its', 'NNS'),
     ('practices', 'VBZ'),
     ('in', 'NN'),
     ('the', 'NN'),
     ('appointment', 'NN'),
     ('of', 'NN'),
     ('appraisers', 'NNS'),
     (',', 'NN'),
     ('guardians', 'NNS'),
     ('and', 'NN'),
     ('administrators', 'NNS'),
     ('and', 'NN'),
     ('the', 'NN'),
     ('awarding', 'VBG'),
     ('of', 'NN'),
     ('fees', 'VBZ'),
     ('and', 'NN'),
     ('compensation', 'NN'),
     ('.', 'NN'),
     ('Wards', 'NNS'),
     ('protected', 'VBD'),
     ('The', 'NN'),
     ('jury', 'NN'),
     ('said', 'NN'),
     ('it', 'NN'),
     ('found', 'NN'),
     ('the', 'NN'),
     ('court', 'NN'),
     ('``', 'NN'),
     ('has', 'NNS'),
     ('incorporated', 'VBD'),
     ('into', 'NN'),
     ('its', 'NNS'),
     ('operating', 'VBG'),
     ('procedures', 'VBZ'),
     ('the', 'NN'),
     ('recommendations', 'NNS'),
     ("''", 'NN'),
     ('of', 'NN'),
     ('two', 'NN'),
     ('previous', 'NNS'),
     ('grand', 'NN'),
     ('juries', 'VBZ'),
     (',', 'NN'),
     ('the', 'NN'),
     ('Atlanta', 'NN'),
     ('Bar', 'NN'),
     ('Association', 'NN'),
     ('and', 'NN'),
     ('an', 'NN'),
     ('interim', 'NN'),
     ('citizens', 'NNS'),
     ('committee', 'NN'),
     ('.', 'NN'),
     ('``', 'NN')




<b>limit_output extension: Maximum message size of 10000 exceeded with 17812 characters</b>


#### The Lookup Tagger


```python
fd = nltk.FreqDist(brown.words(categories='news'))
cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
#most_freq_words = fd.keys()[:100]
fd
```




    FreqDist({'compassion': 1,
              'southpaw': 5,
              'Thakhek': 1,
              'expense': 7,
              'two-family': 1,
              'fine': 17,
              'creature': 2,
              'blonde': 1,
              'Lemon': 3,
              'Rob': 1,
              'KKK': 1,
              'Zone': 1,
              'decent': 2,
              'companies': 18,
              "O'Clock": 1,
              "Emperor's": 1,
              'utility': 5,
              'gruonded': 1,
              'Latin': 7,
              'lay-offs': 4,
              '1.5': 1,
              '$125': 1,
              '3-run': 1,
              'Pye': 1,
              'Mark': 3,
              'seven-hit': 1,
              'stag': 1,
              'construed': 1,
              'chocolate': 1,
              'kept': 16,
              'room': 17,
              'warbling': 1,
              'Tareytown': 1,
              'tour': 7,
              'intruders': 1,
              'Displayed': 1,
              'Comedian': 1,
              '$450': 2,
              'growing': 5,
              'topics': 2,
              'narcotic': 2,
              'multi-family': 1,
              'explaining': 1,
              'hurtling': 1,
              'wood': 1,
              'exposed': 2,
              'stars': 1,
              'strong': 13,
              'Sees': 1,
              'via': 2,
              'remarkably': 1,
              'People': 4,
              'intervening': 1,
              'welcome': 4,
              'President': 89,
              '1920s': 1,
              'jangling': 1,
              'contend': 2,
              'listening': 2,
              'sidelines': 1,
              'know': 28,
              'emissaries': 1,
              'principle': 4,
              'aerial': 3,
              'ruling': 6,
              'do': 63,
              'pull': 2,
              'second-degree': 2,
              'marketed': 1,
              'search': 3,
              'attends': 2,
              'possibly': 3,
              'succession': 1,
              'oil': 6,
              'biennial': 1,
              'Church': 17,
              'perturbed': 1,
              'cent': 51,
              'Just': 7,
              'gangling': 1,
              'its': 174,
              'Price': 3,
              'headed': 13,
              'ova': 1,
              'Pratt': 6,
              'saluted': 1,
              'Ct.': 1,
              'North': 29,
              'filing': 3,
              'Las': 1,
              'percentage': 4,
              'Fulbright': 1,
              'batch': 1,
              'prominently': 1,
              'quality': 3,
              'sound': 4,
              'earned-run': 1,
              'tacked': 1,
              'example': 15,
              'integrated': 2,
              'Margaret': 3,
              "Willie's": 4,
              '101b': 1,
              'documents': 1,
              'Romantic': 1,
              'reading': 9,
              'came': 41,
              'future': 25,
              '114': 1,
              'hitters': 1,
              'Do': 1,
              'plants': 5,
              'Germany': 6,
              'spends': 2,
              'times': 18,
              'Messrs': 1,
              'Administration': 10,
              'Saul': 1,
              'Mongolia': 1,
              'President-elect': 5,
              'Tyson': 1,
              'lows': 1,
              'housed': 1,
              'down': 50,
              'athletics': 1,
              'Santa': 6,
              'Freida': 1,
              'device': 1,
              'grounder': 1,
              'undisputed': 1,
              'like': 46,
              'Paul-Minneapolis': 1,
              'Benched': 2,
              'settled': 4,
              "Caltech's": 1,
              'triumph': 4,
              '53-year-old': 1,
              'earliest': 1,
              'put': 27,
              'defeat': 1,
              'unfair': 2,
              'apparently': 12,
              'unusual': 3,
              'naturalized': 1,
              'Moss': 3,
              'soloists': 1,
              'confrontation': 2,
              'incredible': 2,
              'cleaner': 1,
              'shrugged': 1,
              "moment's": 1,
              '149': 1,
              'paying': 7,
              'Lex': 1,
              'phase': 3,
              'ineptness': 1,
              'plead': 1,
              'Precise': 1,
              'relatives': 2,
              'Study': 1,
              'ninety-nine': 1,
              'bombing': 1,
              'designated': 3,
              'Why': 5,
              'Sent': 1,
              'inlaid': 1,
              'far-reaching': 2,
              'profound': 1,
              'younger': 4,
              '72': 3,
              'Shore': 1,
              'musicians': 3,
              'afraid': 3,
              'granted': 5,
              'illusory': 1,
              'aunts': 1,
              'mostly': 4,
              '31978': 1,
              'confrontations': 1,
              'explicit': 1,
              'underestimate': 1,
              'surviving': 1,
              'enjoys': 2,
              'accidentally': 1,
              'walnut': 1,
              'Foxx': 1,
              'Young': 6,
              'indifference': 1,
              'main': 4,
              'survive': 1,
              'tangible': 1,
              'Stoll': 1,
              'lending': 2,
              'gardenias': 1,
              "Christine's": 1,
              'radioactive': 1,
              'Pilots': 1,
              'Sir': 2,
              'Wolverton': 1,
              'Philip': 2,
              'navy': 2,
              'hailed': 2,
              'How': 5,
              'chart': 2,
              'prepares': 1,
              'sounded': 1,
              'murdered': 1,
              'rehearsal': 1,
              'Pfohl': 3,
              'cross-section': 1,
              "President's": 11,
              'requirement': 4,
              'hunting': 2,
              'Virgin': 2,
              'furs': 1,
              'Halleck': 2,
              'groups': 13,
              'chambers': 2,
              'lie': 1,
              'firm': 31,
              'Denver-area': 1,
              "Patrick's": 2,
              'cash': 7,
              "taxpayer's": 1,
              'Wellesley': 2,
              'arrive': 5,
              'Giorgio': 3,
              'modest': 7,
              'colonialist': 1,
              'Flowers': 2,
              'Turk': 1,
              "Navy's": 2,
              '1926': 1,
              'Brandeis': 1,
              'subdivision': 2,
              'decision': 13,
              'easier': 1,
              'Odell': 1,
              'protested': 1,
              'Haaek': 1,
              'Bessie': 1,
              'conferees': 1,
              'Aquinas': 1,
              'golfer': 2,
              'popularity': 1,
              "Nugent's": 1,
              'couple': 13,
              'Len': 1,
              'signaled': 1,
              'returning': 3,
              'contained': 2,
              'great': 30,
              'Madison': 3,
              'Norell': 2,
              'retire': 2,
              'positive': 2,
              'affect': 3,
              'discredit': 1,
              'burns': 5,
              'regular': 4,
              '11-5': 1,
              'founder': 1,
              'Ruth': 6,
              'Maryland': 5,
              'incumbent': 1,
              'sense': 13,
              'alert': 1,
              'need': 31,
              'folks': 2,
              'P.M.': 2,
              'surveyed': 2,
              'pin': 1,
              'reconsider': 2,
              'an': 300,
              'Buchanan': 1,
              'Perennian': 2,
              'Writers': 1,
              'Vieth': 3,
              'vindicated': 1,
              'hearsay': 1,
              'Institute': 5,
              'decide': 3,
              'bid': 8,
              'bet': 2,
              'word': 14,
              'Award': 7,
              'Vladilen': 1,
              'Prentice-Hall': 1,
              'nationalism': 2,
              'adjustments': 2,
              'troubled': 1,
              'Mullen': 2,
              'motel-keeping': 1,
              'privileges': 3,
              'reveal': 1,
              'politics': 3,
              'bring': 18,
              'Episcopal': 1,
              "Mississippi's": 2,
              'Lavaughn': 1,
              'Hill': 13,
              'guard': 4,
              'very': 33,
              'Besset': 1,
              'disarmament': 2,
              'term': 13,
              'Order': 3,
              'Adjusted': 1,
              'Show': 2,
              'Delinquency': 2,
              'pleasant': 1,
              'covers': 4,
              'original': 6,
              '$10,000': 5,
              'Vandiver': 4,
              'designs': 9,
              'obligated': 1,
              'co-ops': 1,
              'participation': 6,
              'bubble': 1,
              'color': 11,
              'destroy': 5,
              'quota': 1,
              "superintendent's": 1,
              'Perkins': 2,
              '$1,000': 2,
              'Cynthia': 1,
              'harmless': 1,
              'unheard': 1,
              'generated': 1,
              'Lint': 1,
              'Elementary': 2,
              'attract': 2,
              'chest': 2,
              'Indicating': 1,
              'USN.': 1,
              'door-to-door': 1,
              'Opelika': 1,
              '$300,000,000': 1,
              'almost': 24,
              'trials': 1,
              'swine': 1,
              'seldom': 3,
              'Beverly': 7,
              'reads': 2,
              'concerned': 12,
              'Gee': 2,
              'caskets': 1,
              'increasing': 2,
              'Speaker': 3,
              'bond': 11,
              'miracles': 1,
              'message': 7,
              'invoking': 2,
              'Its': 4,
              'Moving': 1,
              'happy': 12,
              'Vague': 1,
              'Ind.': 1,
              'instead': 9,
              'staged': 5,
              'pool': 2,
              'Princes': 1,
              'border': 2,
              'establishing': 3,
              'Berteros': 1,
              'far-flung': 1,
              'Italian': 6,
              'drama': 2,
              'rally': 3,
              'ladies': 3,
              'ranging': 3,
              'carrying': 4,
              '90': 2,
              'concessionaires': 1,
              'Couve': 1,
              'frequent': 2,
              'Trade': 3,
              '7.5': 1,
              '$40,000': 1,
              'menaced': 1,
              'Cen-Tennial': 2,
              'flat': 5,
              'acceptable': 3,
              'Virdon': 2,
              'ugly': 1,
              'Sarmi': 1,
              'cornerstone': 1,
              'Busch': 2,
              'grimly': 1,
              'three-inning': 1,
              'Kaiser': 1,
              'arrange': 2,
              'gesture': 2,
              'letterman': 1,
              'Players': 1,
              'safely': 2,
              'coat': 2,
              "Boston's": 1,
              'innumerable': 1,
              'lawyers': 3,
              'baseballs': 1,
              'turn': 18,
              'Steelers': 2,
              'operating': 4,
              'Parsons': 3,
              'court-appointed': 1,
              'nine-game': 1,
              'tonight': 7,
              'Elsie': 1,
              'last-round': 1,
              'presented': 15,
              'requested': 2,
              'contingency': 1,
              'sta




<b>limit_output extension: Maximum message size of 10000 exceeded with 24097 characters</b>



```python
cfd
```




    ConditionalFreqDist(nltk.probability.FreqDist,
                        {'compassion': FreqDist({'NN': 1}),
                         'southpaw': FreqDist({'NN': 5}),
                         'Thakhek': FreqDist({'NP': 1}),
                         'expense': FreqDist({'NN': 7}),
                         'two-family': FreqDist({'JJ': 1}),
                         'fine': FreqDist({'JJ': 12, 'NN': 4, 'RB': 1}),
                         'creature': FreqDist({'NN': 2}),
                         'blonde': FreqDist({'JJ': 1}),
                         'Lemon': FreqDist({'NP': 3}),
                         'Rob': FreqDist({'NP': 1}),
                         'KKK': FreqDist({'NN': 1}),
                         'Zone': FreqDist({'NN-TL': 1}),
                         'decent': FreqDist({'JJ': 2}),
                         'companies': FreqDist({'NNS': 18}),
                         "O'Clock": FreqDist({'RB-TL': 1}),
                         "Emperor's": FreqDist({'NN$-TL': 1}),
                         'utility': FreqDist({'NN': 5}),
                         'gruonded': FreqDist({'VBD': 1}),
                         'Latin': FreqDist({'JJ': 2, 'JJ-TL': 5}),
                         'lay-offs': FreqDist({'NNS': 3, 'NNS-HL': 1}),
                         '1.5': FreqDist({'CD': 1}),
                         '$125': FreqDist({'NNS': 1}),
                         '3-run': FreqDist({'JJ': 1}),
                         'Pye': FreqDist({'NP': 1}),
                         'Mark': FreqDist({'NP': 3}),
                         'seven-hit': FreqDist({'JJ': 1}),
                         'stag': FreqDist({'NN': 1}),
                         'construed': FreqDist({'VBD': 1}),
                         'chocolate': FreqDist({'NN': 1}),
                         'kept': FreqDist({'VBD': 11, 'VBD-HL': 1, 'VBN': 4}),
                         'room': FreqDist({'NN': 17}),
                         'warbling': FreqDist({'VBG': 1}),
                         'Tareytown': FreqDist({'NP-TL': 1}),
                         'tour': FreqDist({'NN': 7}),
                         'intruders': FreqDist({'NNS': 1}),
                         'Displayed': FreqDist({'VBN': 1}),
                         'Comedian': FreqDist({'NN-TL': 1}),
                         '$450': FreqDist({'NNS': 2}),
                         'growing': FreqDist({'VBG': 5}),
                         'topics': FreqDist({'NNS': 2}),
                         'narcotic': FreqDist({'JJ': 1, 'NN': 1}),
                         'multi-family': FreqDist({'JJ': 1}),
                         'explaining': FreqDist({'VBG': 1}),
                         'hurtling': FreqDist({'VBG': 1}),
                         'wood': FreqDist({'NN': 1}),
                         'exposed': FreqDist({'VBD': 1, 'VBN': 1}),
                         'stars': FreqDist({'NNS': 1}),
                         'strong': FreqDist({'JJ': 13}),
                         'Sees': FreqDist({'VBZ-HL': 1}),
                         'via': FreqDist({'IN': 2}),
                         'remarkably': FreqDist({'QL': 1}),
                         'People': FreqDist({'NNS': 2, 'NNS-TL': 2}),
                         'intervening': FreqDist({'VBG': 1}),
                         'welcome': FreqDist({'JJ': 1, 'NN': 1, 'VB': 2}),
                         'President': FreqDist({'NN': 1, 'NN-TL': 88}),
                         '1920s': FreqDist({'NNS': 1}),
                         'jangling': FreqDist({'VBG': 1}),
                         'contend': FreqDist({'VB': 2}),
                         'listening': FreqDist({'VBG': 2}),
                         'sidelines': FreqDist({'NNS': 1}),
                         'know': FreqDist({'VB': 26, 'VB-HL': 2}),
                         'emissaries': FreqDist({'NNS': 1}),
                         'principle': FreqDist({'NN': 4}),
                         'aerial': FreqDist({'JJ': 3}),
                         'ruling': FreqDist({'NN': 6}),
                         'do': FreqDist({'DO': 62, 'DO-HL': 1}),
                         'pull': FreqDist({'NN': 1, 'VB': 1}),
                         'second-degree': FreqDist({'NN': 2}),
                         'marketed': FreqDist({'VBN': 1}),
                         'search': FreqDist({'NN': 2, 'VB': 1}),
                         'attends': FreqDist({'VBZ': 2}),
                         'possibly': FreqDist({'RB': 3}),
                         'succession': FreqDist({'NN': 1}),
                         'oil': FreqDist({'NN': 6}),
                         'biennial': FreqDist({'JJ': 1}),
                         'Church': FreqDist({'NN-HL': 1, 'NN-TL': 16}),
                         'perturbed': FreqDist({'VBN': 1}),
                         'cent': FreqDist({'NN': 51}),
                         'Just': FreqDist({'RB': 7}),
                         'gangling': FreqDist({'JJ': 1}),
                         'its': FreqDist({'PP$': 174}),
                         'Price': FreqDist({'NN': 1, 'NP': 2}),
                         'headed': FreqDist({'VBD': 6, 'VBN': 7}),
                         'ova': FreqDist({'NN-NC': 1}),
                         'Pratt': FreqDist({'NP': 4, 'NP-TL': 2}),
                         'saluted': FreqDist({'VBN': 1}),
                         'Ct.': FreqDist({'NN-TL': 1}),
                         'North': FreqDist({'JJ-TL': 28, 'NR-TL': 1}),
                         'filing': FreqDist({'VBG': 3}),
                         'Las': FreqDist({'NP': 1}),
                         'percentage': FreqDist({'NN': 4}),
                         'Fulbright': FreqDist({'NP': 1}),
                         'batch': FreqDist({'NN': 1}),
                         'prominently': FreqDist({'RB': 1}),
                         'quality': FreqDist({'NN': 3}),
                         'sound': FreqDist({'JJ': 4}),
                         'earned-run': FreqDist({'NN': 1}),
                         'tacked': FreqDist({'VBD': 1}),
                         'example': FreqDist({'NN': 15}),
                         'integrated': FreqDist({'VBN': 2}),
                         'Margaret': FreqDist({'NP': 3}),
                         "Willie's": FreqDist({'NP$': 4}),
                         '101b': FreqDist({'CD-TL': 1}),
                         'documents': FreqDist({'NNS': 1}),
                         'Romantic': FreqDist({'JJ': 1}),
                         'reading': FreqDist({'NN': 5, 'NN-HL': 1, 'VBG': 3}),
                         'came': FreqDist({'VBD': 41}),
                         'future': FreqDist({'JJ': 8, 'NN': 16, 'NN-HL': 1}),
                         '114': FreqDist({'CD': 1}),
                         'hitters': FreqDist({'NNS': 1}),
                         'Do': FreqDist({'DO': 1}),
                         'plants': FreqDist({'NNS': 5}),
                         'Germany': FreqDist({'NP': 3, 'NP-TL': 3}),
                         'spends': FreqDist({'VBZ': 2}),
                         'times': FreqDist({'NNS': 18}),
                         'Messrs': FreqDist({'NN': 1}),
                         'Administration': FreqDist({'NN-TL': 10}),
                         'Saul': FreqDist({'NP': 1}),
                         'Mongolia': FreqDist({'NP-TL': 1}),
                         'President-elect': FreqDist({'NN-TL': 5}),
                         'Tyson': FreqDist({'NP': 1}),
                         'lows': FreqDist({'NNS': 1}),
                         'housed': FreqDist({'VBN': 1}),
                         'down': FreqDist({'IN': 9, 'RP': 40, 'RP-HL': 1}),
                         'athletics': FreqDist({'NN': 1}),
                         'Santa': FreqDist({'NP': 5, 'NP-HL': 1}),
                         'Freida': FreqDist({'NP': 1}),
                         'device': FreqDist({'NN': 1}),
                         'grounder': FreqDist({'NN': 1}),
                         'undisputed': FreqDist({'JJ': 1}),
                         'like': FreqDist({'CS': 26,
                                   'IN': 1,
                                   'JJ': 1,
                                   'VB': 17,
                                   'VB-HL': 1}),
                         'Paul-Minneapolis': FreqDist({'NP-HL': 1}),
                         'Benched': FreqDist({'VBN': 2}),
                         'settled': FreqDist({'VBN': 4}),
                         "Caltech's": FreqDist({'NP$': 1}),
                         'triumph': FreqDist({'NN': 4}),
                         '53-year-old': FreqDist({'JJ': 1}),
                         'earliest': FreqDist({'JJT': 1}),
                         'put': FreqDist({'VB': 8, 'VBD': 14, 'VBN': 5}),
                         'defeat': FreqDist({'NN': 1}),
                         'unfair': FreqDist({'JJ': 2}),
                         'apparently': FreqDist({'RB': 12}),
                         'unusual': FreqDist({'JJ': 3}),
                         'naturalized': FreqDist({'VBN': 1}),
                         'Moss': FreqDist({'NP': 2, 'NP-TL': 1}),
                         'soloists': FreqDist({'NNS': 1}),
                         'confrontation': FreqDist({'NN': 2}),
                         'incredible': FreqDist({'JJ': 2}),
                         'cleaner': FreqDist({'JJR': 1}),
                         'shrugged': FreqDist({'VBD': 1}),
                         "moment's": FreqDist({'NN$': 1}),
                         '149': FreqDist({'CD': 1}),
                         'paying': FreqDist({'VBG': 7}),
                         'Lex': FreqDist({'NP': 1}),
                         'phase': FreqDist({'NN': 3}),
                         'ineptness': FreqDist({'NN': 1}),
                         'plead': FreqDist({'VB': 1}),
                         'Precise': FreqDist({'JJ-HL': 1}),
                         'relatives': FreqDist({'NNS': 2}),
                         'Study': FreqDist({'VB-TL': 1}),
                         'ninety-nine': FreqDist({'CD': 1}),
                         'bombing': FreqDist({'NN': 1}),
                         'designated': FreqDist({'VBN': 3}),
                         'Why': FreqDist({'WRB': 5}),
                         'Sent': FreqDist({'VBN-HL': 1}),
                         'inlaid': FreqDist({'VBN': 1}),
                         'far-reaching': FreqDist({'JJ': 2}),
                         'profound': FreqDist({'JJ': 1}),
                         'younger': FreqDist({'JJR': 4}),
                         '72': FreqDist({'CD': 3}),
                         'Shore': FreqDist({'NN-TL': 1}),
                         'musicians': FreqDist({'NNS': 3}),
                         'afraid': FreqDist({'JJ': 3}),
                         'granted': FreqDist({'VBN': 5}),
                         'illusory': FreqDist({'JJ': 1}),
                         'aunts': FreqDist({'NNS': 1}),
              




<b>limit_output extension: Maximum message size of 10000 exceeded with 55663 characters</b>


#### Unigram Tagging (no context)


```python
from nltk.corpus import brown
brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')
unigram_tagger = nltk.UnigramTagger(brown_tagged_sents) #Training 
unigram_tagger.tag(brown_sents[2007])
```




    [('Various', 'JJ'),
     ('of', 'IN'),
     ('the', 'AT'),
     ('apartments', 'NNS'),
     ('are', 'BER'),
     ('of', 'IN'),
     ('the', 'AT'),
     ('terrace', 'NN'),
     ('type', 'NN'),
     (',', ','),
     ('being', 'BEG'),
     ('on', 'IN'),
     ('the', 'AT'),
     ('ground', 'NN'),
     ('floor', 'NN'),
     ('so', 'QL'),
     ('that', 'CS'),
     ('entrance', 'NN'),
     ('is', 'BEZ'),
     ('direct', 'JJ'),
     ('.', '.')]



#### N-gram tagger


```python
bigram_tagger = nltk.BigramTagger(brown_tagged_sents)
bigram_tagger.tag(brown_sents[2007])
```




    [('Various', 'JJ'),
     ('of', 'IN'),
     ('the', 'AT'),
     ('apartments', 'NNS'),
     ('are', 'BER'),
     ('of', 'IN'),
     ('the', 'AT'),
     ('terrace', 'NN'),
     ('type', 'NN'),
     (',', ','),
     ('being', 'BEG'),
     ('on', 'IN'),
     ('the', 'AT'),
     ('ground', 'NN'),
     ('floor', 'NN'),
     ('so', 'CS'),
     ('that', 'CS'),
     ('entrance', 'NN'),
     ('is', 'BEZ'),
     ('direct', 'JJ'),
     ('.', '.')]



这样有个问题，如果tag的句子中的某个词的context在训练集里面没有，哪怕这个词在训练集中有，也无法进行标注，还是要通过`backoff`来解决这样的问题



```python
t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(brown_tagged_sents, backoff=t0)
t2 = nltk.BigramTagger(brown_tagged_sents, backoff=t1)
t2.tag(brown_sents[2007])
```




    [('Various', 'JJ'),
     ('of', 'IN'),
     ('the', 'AT'),
     ('apartments', 'NNS'),
     ('are', 'BER'),
     ('of', 'IN'),
     ('the', 'AT'),
     ('terrace', 'NN'),
     ('type', 'NN'),
     (',', ','),
     ('being', 'BEG'),
     ('on', 'IN'),
     ('the', 'AT'),
     ('ground', 'NN'),
     ('floor', 'NN'),
     ('so', 'CS'),
     ('that', 'CS'),
     ('entrance', 'NN'),
     ('is', 'BEZ'),
     ('direct', 'JJ'),
     ('.', '.')]



n-gram tagger存在的问题是:
- model会占用比较大的空间
- 还有就是在考虑context时，只会考虑前面词的tag，而不会考虑词本身



#### Brill tagging

用存储rule来代替model，这样可以节省大量的空间，同时在rule中不限制仅考虑tag，也可以考虑word本身

例子:

(1) replace NN with VB when the previous word is TO;

(2) replace TO with IN when the next tag is NNS.


![](../../images/19.png)


第一步用unigram tagger对所有词做一遍tagging，这里面可能有很多不准确的

下面就用rule来纠正第一步中guess错的那些词的tag，最终得到比较准确的tagging

> 那么这些rules是怎么生成的?

在training阶段自动生成的: 

During its training phase, the tagger guesses values for T1, T2, and C, to create thousands of candidate rules. Each rule is scored according to its net benefit: the number of incorrect tags that it corrects, less the number
of correct tags it incorrectly modifies.

----

rules的例子:

- NN -> VB if the tag of the preceding word is 'TO'
- NN -> VBD if the tag of the following word is 'DT'
- NN -> VBD if the tag of the preceding word is 'NNS'
- NN -> NNP if the tag of words i-2...i-1 is '-NONE-'
- NN -> NNP if the tag of the following word is 'NNP'
- NN -> NNP if the text of words i-2...i-1 is 'like'
- NN -> VBN if the text of the following word is '*-1'


----




### Token normalization

![](../../images/20.png)

![](../../images/21.png)


```python
import nltk
text1 = 'feet, cats, wolves, talked'
text2 = 'feet cats wolves talked'
tokenizer = nltk.tokenize.TreebankWordTokenizer()
tokens1 = tokenizer.tokenize(text1)
tokens2 = tokenizer.tokenize(text2)
tokens1, tokens2
```




    (['feet', ',', 'cats', ',', 'wolves', ',', 'talked'],
     ['feet', 'cats', 'wolves', 'talked'])




```python
stemmer = nltk.stem.PorterStemmer()
" ".join(stemmer.stem(token) for token in tokens1)
```




    'feet , cat , wolv , talk'



## Feature extraction from text

### BOW

![](../../images/22.png)

> how to preserve some order info?

![](../../images/23.png)


![](../../images/25.png)

> Question 


![](../../images/24.png)





### TF-IDF (词频-逆文件频率)

是一种用于资讯检索与资讯探勘的常用加权技术。TF-IDF是一种统计方法，用以评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度。字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。

一个词语在一篇文章中出现次数越多, 同时在所有文档中出现次数越少, 越能够代表该文章



#### TF (Term Frequency)

![](../../images/29.png)

#### IDF (Inverse document frequency)

- $N = |D|$ : total number of documents in corpus
- $\mid {d \in D: t \in d} \mid$: number of documents where term $t$ appears
- $idf(t, D) = \log \frac{N}{\mid {d \in D: t \in d} \mid}$

#### TF-IDF

$$tfidf(t,d,D) = tf(t,d) \cdot idf(t,D)$$
- A high weight if TF-IDF is reached by a __high term frequency (TF)__ in the given document and __a low document frequency of the term (IDF)__ in the whole


![](../../images/28.png)




```python
import numpy as np
import pandas as pd
import csv
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
DIR = "./all/"


def load_train_data(skip_content=False):
    categories = ['cooking', 'robotics', 'travel', 'crypto', 'diy', 'biology']
    train_data = []
    for cat in categories:
        if skip_content:
            data = pd.read_csv("{}{}.csv".format(DIR, cat), usecols=['id', 'title', 'tags'])
        else:
            data = pd.read_csv("{}{}.csv".format(DIR, cat))
        data['category'] = cat
        train_data.append(data)
    
    return pd.concat(train_data)
load_train_data()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>title</th>
      <th>content</th>
      <th>tags</th>
      <th>category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>How can I get chewy chocolate chip cookies?</td>
      <td>&lt;p&gt;My chocolate chips cookies are always too c...</td>
      <td>baking cookies texture</td>
      <td>cooking</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>How should I cook bacon in an oven?</td>
      <td>&lt;p&gt;I've heard of people cooking bacon in an ov...</td>
      <td>oven cooking-time bacon</td>
      <td>cooking</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>What is the difference between white and brown...</td>
      <td>&lt;p&gt;I always use brown extra large eggs, but I ...</td>
      <td>eggs</td>
      <td>cooking</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>What is the difference between baking soda and...</td>
      <td>&lt;p&gt;And can I use one in place of the other in ...</td>
      <td>substitutions please-remove-this-tag baking-so...</td>
      <td>cooking</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>In a tomato sauce recipe, how can I cut the ac...</td>
      <td>&lt;p&gt;It seems that every time I make a tomato sa...</td>
      <td>sauce pasta tomatoes italian-cuisine</td>
      <td>cooking</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>What ingredients (available in specific region...</td>
      <td>&lt;p&gt;I have a recipe that calls for fresh parsle...</td>
      <td>substitutions herbs parsley</td>
      <td>cooking</td>
    </tr>
    <tr>
      <th>6</th>
      <td>9</td>
      <td>What is the internal temperature a steak shoul...</td>
      <td>&lt;p&gt;I'd like to know when to take my steaks off...</td>
      <td>food-safety beef cooking-time</td>
      <td>cooking</td>
    </tr>
    <tr>
      <th>7</th>
      <td>11</td>
      <td>How should I poach an egg?</td>
      <td>&lt;p&gt;What's the best method to poach an egg with...</td>
      <td>eggs basics poaching</td>
      <td>cooking</td>
    </tr>
    <tr>
      <th>8</th>
      <td>12</td>
      <td>How can I make my Ice Cream "creamier"</td>
      <td>&lt;p&gt;My ice cream doesn't feel creamy enough.  I...</td>
      <td>ice-cream</td>
      <td>cooking</td>
    </tr>
    <tr>
      <th>9</th>
      <td>17</td>
      <td>How long and at what temperature do the variou...</td>
      <td>&lt;p&gt;I'm interested in baking thighs, legs, brea...</td>
      <td>baking chicken cooking-time</td>
      <td>cooking</td>
    </tr>
    <tr>
      <th>10</th>
      <td>23</td>
      <td>Besides salmon, what other meats can be grille...</td>
      <td>&lt;p&gt;I've fallen in love with this wonderful &lt;a ...</td>
      <td>grilling salmon cedar-plank</td>
      <td>cooking</td>
    </tr>
    <tr>
      <th>11</th>
      <td>27</td>
      <td>Do I need to sift flour that is labeled sifted?</td>
      <td>&lt;p&gt;Is there really an advantage to sifting flo...</td>
      <td>baking flour measurements sifting</td>
      <td>cooking</td>
    </tr>
    <tr>
      <th>12</th>
      <td>28</td>
      <td>Storage life for goose fat</td>
      <td>&lt;p&gt;When I roast a goose, I decant the fat, str...</td>
      <td>storage-method storage-lifetime fats</td>
      <td>cooking</td>
    </tr>
    <tr>
      <th>13</th>
      <td>30</td>
      <td>Pressure canning instructions</td>
      <td>&lt;p&gt;Where can safe and reliable instructions (i...</td>
      <td>canning pressure-canner food-preservation</td>
      <td>cooking</td>
    </tr>
    <tr>
      <th>14</th>
      <td>32</td>
      <td>What's a good resource for knowing what spices...</td>
      <td>&lt;p&gt;I know what spices like garlic and black pe...</td>
      <td>spices resources basics learning</td>
      <td>cooking</td>
    </tr>
    <tr>
      <th>15</th>
      <td>36</td>
      <td>Is it safe to leave butter at room temperature?</td>
      <td>&lt;p&gt;Is it safe to leave butter at room temperat...</td>
      <td>food-safety storage-method storage-lifetime bu...</td>
      <td>cooking</td>
    </tr>
    <tr>
      <th>16</th>
      <td>38</td>
      <td>Does resting the dough for a long time reduce ...</td>
      <td>&lt;p&gt;In this &lt;a href="http://www.chefmichaelsmit...</td>
      <td>baking bread dough</td>
      <td>cooking</td>
    </tr>
    <tr>
      <th>17</th>
      <td>54</td>
      <td>How should I prepare Risotto</td>
      <td>&lt;p&gt;I've been watching a lot of Hells Kitchen, ...</td>
      <td>rice italian-cuisine risotto</td>
      <td>cooking</td>
    </tr>
    <tr>
      <th>18</th>
      <td>57</td>
      <td>How does a splash of vinegar help when poachin...</td>
      <td>&lt;p&gt;What does splashing in a shot of white vine...</td>
      <td>eggs food-science vinegar poaching</td>
      <td>cooking</td>
    </tr>
    <tr>
      <th>19</th>
      <td>61</td>
      <td>What are the pros and cons of storing bread in...</td>
      <td>&lt;p&gt;Why should/shouldn't I store my bread in th...</td>
      <td>storage-method bread</td>
      <td>cooking</td>
    </tr>
    <tr>
      <th>20</th>
      <td>62</td>
      <td>What are some good resources for learning Knif...</td>
      <td>&lt;p&gt;What are some good resources for learning k...</td>
      <td>knife-skills resources learning cutting</td>
      <td>cooking</td>
    </tr>
    <tr>
      <th>21</th>
      <td>66</td>
      <td>How to calculate the calorie content of cooked...</td>
      <td>&lt;p&gt;I like to cook from scratch, and I'm curren...</td>
      <td>nutrient-composition calories</td>
      <td>cooking</td>
    </tr>
    <tr>
      <th>22</th>
      <td>68</td>
      <td>Recommendations for spice organization strategies</td>
      <td>&lt;p&gt;Spices have always been the hardest thing f...</td>
      <td>storage-method spices organization pantry</td>
      <td>cooking</td>
    </tr>
    <tr>
      <th>23</th>
      <td>70</td>
      <td>Shelf life of spices</td>
      <td>&lt;p&gt;The common wisdom I've heard is that dried ...</td>
      <td>storage-method storage-lifetime spices</td>
      <td>cooking</td>
    </tr>
    <tr>
      <th>24</th>
      <td>76</td>
      <td>How do I convert between the various measureme...</td>
      <td>&lt;p&gt;I found a recipe that's using one or more m...</td>
      <td>conversion measurements</td>
      <td>cooking</td>
    </tr>
    <tr>
      <th>25</th>
      <td>81</td>
      <td>How do I pound chicken (or other meat) without...</td>
      <td>&lt;p&gt;Despite my best efforts, my kitchen (and so...</td>
      <td>chicken meat chicken-breast tenderizing</td>
      <td>cooking</td>
    </tr>
    <tr>
      <th>26</th>
      <td>84</td>
      <td>Is there a formula for converting pancake batt...</td>
      <td>&lt;p&gt;I have a wonderful pancake recipe that I wo...</td>
      <td>baking pancakes conversion waffle</td>
      <td>cooking</td>
    </tr>
    <tr>
      <th>27</th>
      <td>85</td>
      <td>Wok preparation and caring</td>
      <td>&lt;p&gt;What is a good technique for initially seas...</td>
      <td>equipment wok seasoning-pans</td>
      <td>cooking</td>
    </tr>
    <tr>
      <th>28</th>
      <td>87</td>
      <td>How can I keep delicate food from sticking to ...</td>
      <td>&lt;p&gt;When I grill fish or chicken, often much of...</td>
      <td>grilling</td>
      <td>cooking</td>
    </tr>
    <tr>
      <th>29</th>
      <td>89</td>
      <td>What can I do to help my avocados ripen?</td>
      <td>&lt;p&gt;I bought some avocados recently, and one of...</td>
      <td>storage-method ripe avocados</td>
      <td>cooking</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>13166</th>
      <td>51207</td>
      <td>Is it possible for any impurities to get throu...</td>
      <td>&lt;p&gt;My parents always told me not to eat from t...</td>
      <td>microbiology health trees dendrology</td>
      <td>biology</td>
    </tr>
    <tr>
      <th>13167</th>
      <td>51208</td>
      <td>Is this a bedbug or a book lice or what?</td>
      <td>&lt;p&gt;So basically i found this thing in my home....</td>
      <td>species-identification entomology</td>
      <td>biology</td>
    </tr>
    <tr>
      <th>13168</th>
      <td>51209</td>
      <td>What happens when the congenitally blind brain...</td>
      <td>&lt;p&gt;If a congenital blind person gets eye sight...</td>
      <td>human-biology neuroscience brain vision human-eye</td>
      <td>biology</td>
    </tr>
    <tr>
      <th>13169</th>
      <td>51212</td>
      <td>Is there a maximum limit of connections betwee...</td>
      <td>&lt;p&gt;As neurons fire and work together, their co...</td>
      <td>neuroscience neuroanatomy neurology</td>
      <td>biology</td>
    </tr>
    <tr>
      <th>13170</th>
      <td>51214</td>
      <td>Why don't onion root tips show mitotic stages?</td>
      <td>&lt;p&gt;It was back in the winter that we did the O...</td>
      <td>botany mitosis</td>
      <td>biology</td>
    </tr>
    <tr>
      <th>13171</th>
      <td>51216</td>
      <td>why there are both antibodies A and B in blood...</td>
      <td>&lt;p&gt;Why the blood group O people have both A an...</td>
      <td>genetics</td>
      <td>biology</td>
    </tr>
    <tr>
      <th>13172</th>
      <td>51218</td>
      <td>Which organs accumulate the highest concentrat...</td>
      <td>&lt;p&gt;&lt;strong&gt;Which organs would process and filt...</td>
      <td>microbiology hematology blood-circulat




<b>limit_output extension: Maximum message size of 10000 exceeded with 15990 characters</b>



```python
def load_test_data():
    test_data = pd.read_csv(DIR + 'test.csv')
    return test_data

test_data = load_test_data()
test_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>title</th>
      <th>content</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>What is spin as it relates to subatomic partic...</td>
      <td>&lt;p&gt;I often hear about subatomic particles havi...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>What is your simplest explanation of the strin...</td>
      <td>&lt;p&gt;How would you explain string theory to non ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Lie theory, Representations and particle physics</td>
      <td>&lt;p&gt;This is a question that has been posted at ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7</td>
      <td>Will Determinism be ever possible?</td>
      <td>&lt;p&gt;What are the main problems that we need to ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9</td>
      <td>Hamilton's Principle</td>
      <td>&lt;p&gt;Hamilton's principle states that a dynamic ...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>13</td>
      <td>What is sound and how is it produced?</td>
      <td>&lt;p&gt;I've been using the term "sound" all my lif...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>15</td>
      <td>What experiment would disprove string theory?</td>
      <td>&lt;p&gt;I know that there's big controversy between...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>17</td>
      <td>Why does the sky change color? Why the sky is ...</td>
      <td>&lt;p&gt;Why does the sky change color? Why the sky ...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>19</td>
      <td>How's the energy of particle collisions calcul...</td>
      <td>&lt;p&gt;Physicists often refer to the energy of col...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>21</td>
      <td>Monte Carlo use</td>
      <td>&lt;p&gt;Where is the Monte Carlo method used in phy...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>24</td>
      <td>Does leaning (banking) help cause turning on a...</td>
      <td>&lt;p&gt;I think it's clear enough that if you turn ...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>26</td>
      <td>Velocity of Object from electromagnetic field</td>
      <td>&lt;p&gt;I am wondering if someone could provide me ...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>27</td>
      <td>What is the difference between a measurement a...</td>
      <td>&lt;p&gt;We've learned that the wave function of a p...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>29</td>
      <td>How to calculate average speed?</td>
      <td>&lt;p&gt;I recently encountered a puzzle where a per...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>31</td>
      <td>Lay explanation of the special theory of relat...</td>
      <td>&lt;p&gt;What is Einstein's theory of &lt;a href="http:...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>32</td>
      <td>How to show that the Coriolis effect is irrele...</td>
      <td>&lt;p&gt;There is a common myth that water flowing o...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>35</td>
      <td>Where do magnets get the energy to repel?</td>
      <td>&lt;p&gt;If I separate two magnets whose opposite po...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>37</td>
      <td>How to check Einstein-like equations on their ...</td>
      <td>&lt;p&gt;Physicists studying the grounds of physics ...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>41</td>
      <td>Impressions of Topological field theories in m...</td>
      <td>&lt;p&gt;There have been recent results in mathemati...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>49</td>
      <td>What is a capacitive screen sensing?</td>
      <td>&lt;p&gt;What should be a properties of a body so a ...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>52</td>
      <td>How do 2 magnets spin by themselves if positio...</td>
      <td>&lt;p&gt;A few years ago I went to a museum, where t...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>62</td>
      <td>Why is the LHC circular and 27km long?</td>
      <td>&lt;p&gt;The LHC in Geneva is a circular accelerator...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>68</td>
      <td>What causes polarised materials to change colo...</td>
      <td>&lt;p&gt;Our physics teacher showed the class a real...</td>
    </tr>
    <tr>
      <th>23</th>
      <td>71</td>
      <td>What is an intuitive explanation of Gouy phase?</td>
      <td>&lt;p&gt;In laser resonators, higher order modes (i....</td>
    </tr>
    <tr>
      <th>24</th>
      <td>72</td>
      <td>Proton therapy in cancer treatment</td>
      <td>&lt;p&gt;Why are protons used in cancer therapy ? Is...</td>
    </tr>
    <tr>
      <th>25</th>
      <td>73</td>
      <td>How do physicists use solutions to the Yang-Ba...</td>
      <td>&lt;p&gt;As a mathematician working the area of repr...</td>
    </tr>
    <tr>
      <th>26</th>
      <td>75</td>
      <td>Mnemonics to remember various properties of ma...</td>
      <td>&lt;p&gt;I'm trying to figure out how to remember th...</td>
    </tr>
    <tr>
      <th>27</th>
      <td>78</td>
      <td>Why do neutrons repel each other?</td>
      <td>&lt;p&gt;I can understand why 2 protons will repel e...</td>
    </tr>
    <tr>
      <th>28</th>
      <td>79</td>
      <td>Is quantum entanglement mediated by an interac...</td>
      <td>&lt;p&gt;You can get two photons entangled, and send...</td>
    </tr>
    <tr>
      <th>29</th>
      <td>83</td>
      <td>How is squeezed light produced?</td>
      <td>&lt;p&gt;Ordinary laser light has equal uncertainty ...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>81896</th>
      <td>278070</td>
      <td>biconvex vs plano convex lenses for 4f imaging</td>
      <td>&lt;p&gt;Will biconvex or plano convex lenses be bes...</td>
    </tr>
    <tr>
      <th>81897</th>
      <td>278071</td>
      <td>Formula of the Magnus force</td>
      <td>&lt;p&gt;I was researching online about the &lt;a href=...</td>
    </tr>
    <tr>
      <th>81898</th>
      <td>278075</td>
      <td>Is this "invention" by a Dutch inventor a hoax?</td>
      <td>&lt;p&gt;I'm from media and would appreciate if you ...</td>
    </tr>
    <tr>
      <th>81899</th>
      <td>278077</td>
      <td>Could you please diagram a few simple Z boson/...</td>
      <td>&lt;p&gt;Could you please diagram a few simple Z bos...</td>
    </tr>
    <tr>
      <th>81900</th>
      <td>278079</td>
      <td>Is F.D.C. Willard's second helium paper availa...</td>
      <td>&lt;p&gt;I am curious to read F.D.C. Willard's secon...</td>
    </tr>
    <tr>
      <th>81901</th>
      <td>278080</td>
      <td>How can we measure the frequency of taste of a...</td>
      <td>&lt;p&gt;How can we measure the frequency of taste f...</td>
    </tr>
    <tr>
      <th>81902</th>
      <td>278081</td>
      <td>The position vector $\mathbf{r}$ in the electr...</td>
      <td>&lt;p&gt;I often see the electric field denoted \beg...</td>
    </tr>
    <tr>
      <th>81903</th>
      <td>278084</td>
      <td>Master-level Minicourse on Topological Propert...</td>
      <td>&lt;p&gt;I am doing a master's thesis on the propert...</td>
    </tr>
    <tr>
      <th>81904</th>
      <td>278086</td>
      <td>Photon Energy and frequency</td>
      <td>&lt;p&gt;If every photon is identical and travels wi...</td>
    </tr>
    <tr>
      <th>81905</th>
      <td>278088</td>
      <td>What would happen if you shot a rocket while i...</td>
      <td>&lt;p&gt;Recently I've been hearing a lot about that...</td>
    </tr>
    <tr>
      <th>81906</th>
      <td>278091</td>
      <td>Difference/relation between Zeeman effect and ...</td>
      <td>&lt;p&gt;From what I see, there are to ways of treat...</td>
    </tr>
    <tr>
      <th>81907</th>
      <td>278092</td>
      <td>Falling Chimney using Lagranges Equation</td>
      <td>&lt;p&gt;There is this interesting problem for a fal...</td>
    </tr>
    <tr>
      <th>81908</th>
      <td>278093</td>
      <td>What is the general form of projection operators?</td>
      <td>&lt;p&gt;Usaully, a projection operator is expressed...</td>
    </tr>
    <tr>
      <th>81909</th>
      <td>278095</td>
      <td>What's the difference between a tachyon and an...</td>
      <td>&lt;p&gt;I'd like to preemptively apologize for bein...</td>
    </tr>
    <tr>
      <th>81910</th>
      <td>278096</td>
      <td>Any equipment that allows the viewing of a las...</td>
      <td>&lt;p&gt;Would equipment such as infrared googles, n...</td>
    </tr>
    <tr>
      <th>81911</th>
      <td>278099</td>
      <td>Calculate electric energy</td>
      <td>&lt;p&gt;An electrically driven train has weight 120...</td>
    </tr>
    <tr>
      <th>81912</th>
      <td>278101</td>
      <td>Creation of electron = creation of mass?</td>
      <td>&lt;p&gt;By turning a turbine for example, if I'm no...</td>
    </tr>
    <tr>
      <th>81913</th>
      <td>278107</td>
      <td>Force needed to lift a hinged slab/lever?</td>
      <td>&lt;p&gt;I'm developing a prop and need to figure ou...</td>
    </tr>
    <tr>
      <th>81914</th>
      <td>278108</td>
      <td>Why are nuclide charts shown two completely di...</td>
      <td>&lt;p&gt;Why are some nuclide charts shown with Z on...</td>
    </tr>
    <tr>
      <th>81915</th>
      <td>278109</td>
      <td>If a gas in a pipe is travelling from a large ...</td>
      <td>&lt;p&gt;So in a problem like the one in the link be...</td>
    </tr>
    <tr>
      <th>81916</th>
      <td>278111</td>
      <td>Can we detect neutrinos (interacti




<b>limit_output extension: Maximum message size of 10000 exceeded with 11865 characters</b>



```python
def merge(row):
    title = row['title']
    content = row['content']
    clean_content = BeautifulSoup(content, "html.parser")
    clean_content = clean_content.get_text()
    row['text'] = title + " " + clean_content
    return row
```


```python
nlp_test_data = test_data.apply(merge, axis=1)[['id', 'text']]
```


```python
nlp_test_data
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>What is spin as it relates to subatomic partic...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>What is your simplest explanation of the strin...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Lie theory, Representations and particle physi...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7</td>
      <td>Will Determinism be ever possible? What are th...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9</td>
      <td>Hamilton's Principle Hamilton's principle stat...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>13</td>
      <td>What is sound and how is it produced? I've bee...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>15</td>
      <td>What experiment would disprove string theory? ...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>17</td>
      <td>Why does the sky change color? Why the sky is ...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>19</td>
      <td>How's the energy of particle collisions calcul...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>21</td>
      <td>Monte Carlo use Where is the Monte Carlo metho...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>24</td>
      <td>Does leaning (banking) help cause turning on a...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>26</td>
      <td>Velocity of Object from electromagnetic field ...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>27</td>
      <td>What is the difference between a measurement a...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>29</td>
      <td>How to calculate average speed? I recently enc...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>31</td>
      <td>Lay explanation of the special theory of relat...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>32</td>
      <td>How to show that the Coriolis effect is irrele...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>35</td>
      <td>Where do magnets get the energy to repel? If I...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>37</td>
      <td>How to check Einstein-like equations on their ...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>41</td>
      <td>Impressions of Topological field theories in m...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>49</td>
      <td>What is a capacitive screen sensing? What shou...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>52</td>
      <td>How do 2 magnets spin by themselves if positio...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>62</td>
      <td>Why is the LHC circular and 27km long? The LHC...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>68</td>
      <td>What causes polarised materials to change colo...</td>
    </tr>
    <tr>
      <th>23</th>
      <td>71</td>
      <td>What is an intuitive explanation of Gouy phase...</td>
    </tr>
    <tr>
      <th>24</th>
      <td>72</td>
      <td>Proton therapy in cancer treatment Why are pro...</td>
    </tr>
    <tr>
      <th>25</th>
      <td>73</td>
      <td>How do physicists use solutions to the Yang-Ba...</td>
    </tr>
    <tr>
      <th>26</th>
      <td>75</td>
      <td>Mnemonics to remember various properties of ma...</td>
    </tr>
    <tr>
      <th>27</th>
      <td>78</td>
      <td>Why do neutrons repel each other? I can unders...</td>
    </tr>
    <tr>
      <th>28</th>
      <td>79</td>
      <td>Is quantum entanglement mediated by an interac...</td>
    </tr>
    <tr>
      <th>29</th>
      <td>83</td>
      <td>How is squeezed light produced? Ordinary laser...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>81896</th>
      <td>278070</td>
      <td>biconvex vs plano convex lenses for 4f imaging...</td>
    </tr>
    <tr>
      <th>81897</th>
      <td>278071</td>
      <td>Formula of the Magnus force I was researching ...</td>
    </tr>
    <tr>
      <th>81898</th>
      <td>278075</td>
      <td>Is this "invention" by a Dutch inventor a hoax...</td>
    </tr>
    <tr>
      <th>81899</th>
      <td>278077</td>
      <td>Could you please diagram a few simple Z boson/...</td>
    </tr>
    <tr>
      <th>81900</th>
      <td>278079</td>
      <td>Is F.D.C. Willard's second helium paper availa...</td>
    </tr>
    <tr>
      <th>81901</th>
      <td>278080</td>
      <td>How can we measure the frequency of taste of a...</td>
    </tr>
    <tr>
      <th>81902</th>
      <td>278081</td>
      <td>The position vector $\mathbf{r}$ in the electr...</td>
    </tr>
    <tr>
      <th>81903</th>
      <td>278084</td>
      <td>Master-level Minicourse on Topological Propert...</td>
    </tr>
    <tr>
      <th>81904</th>
      <td>278086</td>
      <td>Photon Energy and frequency If every photon is...</td>
    </tr>
    <tr>
      <th>81905</th>
      <td>278088</td>
      <td>What would happen if you shot a rocket while i...</td>
    </tr>
    <tr>
      <th>81906</th>
      <td>278091</td>
      <td>Difference/relation between Zeeman effect and ...</td>
    </tr>
    <tr>
      <th>81907</th>
      <td>278092</td>
      <td>Falling Chimney using Lagranges Equation There...</td>
    </tr>
    <tr>
      <th>81908</th>
      <td>278093</td>
      <td>What is the general form of projection operato...</td>
    </tr>
    <tr>
      <th>81909</th>
      <td>278095</td>
      <td>What's the difference between a tachyon and an...</td>
    </tr>
    <tr>
      <th>81910</th>
      <td>278096</td>
      <td>Any equipment that allows the viewing of a las...</td>
    </tr>
    <tr>
      <th>81911</th>
      <td>278099</td>
      <td>Calculate electric energy An electrically driv...</td>
    </tr>
    <tr>
      <th>81912</th>
      <td>278101</td>
      <td>Creation of electron = creation of mass? By tu...</td>
    </tr>
    <tr>
      <th>81913</th>
      <td>278107</td>
      <td>Force needed to lift a hinged slab/lever? I'm ...</td>
    </tr>
    <tr>
      <th>81914</th>
      <td>278108</td>
      <td>Why are nuclide charts shown two completely di...</td>
    </tr>
    <tr>
      <th>81915</th>
      <td>278109</td>
      <td>If a gas in a pipe is travelling from a large ...</td>
    </tr>
    <tr>
      <th>81916</th>
      <td>278111</td>
      <td>Can we detect neutrinos (interaction) in other...</td>
    </tr>
    <tr>
      <th>81917</th>
      <td>278113</td>
      <td>Is there a mass relation of the core left behi...</td>
    </tr>
    <tr>
      <th>81918</th>
      <td>278116</td>
      <td>Vector and frame of reference? My textbook has...</td>
    </tr>
    <tr>
      <th>81919</th>
      <td>278117</td>
      <td>How fast can the Earth spin and support life? ...</td>
    </tr>
    <tr>
      <th>81920</th>
      <td>278118</td>
      <td>Operators on a joint Hilbert Space Can anyone ...</td>
    </tr>
    <tr>
      <th>81921</th>
      <td>278119</td>
      <td>Kinematics (Projectile Motion) A projectile is...</td>
    </tr>
    <tr>
      <th>81922</th>
      <td>278120</td>
      <td>How is lift generated due to Coanda effect I c...</td>
    </tr>
    <tr>
      <th>81923</th>
      <td>278121</td>
      <td>Why is a resonance curve asymmetric? When I wa...</td>
    </tr>
    <tr>
      <th>81924</th>
      <td>278124</td>
      <td>What are the forces acting during a drop impac...</td>
    </tr>
    <tr>
      <th>81925</th>
      <td>278126</td>
      <td>Gravity manipulation i s it a possibly? I have...</td>
    </tr>
  </tbody>
</table>
<p>81926 rows × 2 columns</p>
</div>




```python
tfidf = TfidfVectorizer(analyzer = "word", 
                        max_features = 5000, 
                        stop_words="english", 
                        ngram_range=(1,2))
features = tfidf.fit_transform(nlp_test_data['text']).toarray()
```


```python
pd.DataFrame(features) # very sparse matrix
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>4990</th>
      <th>4991</th>
      <th>4992</th>
      <th>4993</th>
      <th>4994</th>
      <th>4995</th>
      <th>4996</th>
      <th>4997</th>
      <th>4998</th>
      <th>4999</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
    




<b>limit_output extension: Maximum message size of 10000 exceeded with 26541 characters</b>



```python
tfidf_tags = []
top_n = -5

feature_array = np.array(tfidf.get_feature_names())
print(feature_array)
tfidf_sorting = np.argsort(features)
print(tfidf_sorting)

for i, e in enumerate(tfidf_sorting):
    tmp_tags = []
    indexes = e[top_n:]
    for idx in indexes:
        cur_tag = feature_array[idx]
        if features[i][idx] > 0.1 and len(cur_tag)>3 and '_' not in cur_tag:
            tmp_tags.append(cur_tag.replace(' ', '-'))
    tfidf_tags.append(" ".join(tmp_tags))
```

![](../../images/30.png)

![](../../images/31.png)


![](../../images/32.png)

![](../../images/33.png)

![](../../images/34.png)





### Hashing Example

![](../../images/35.png)

![](../../images/36.png)








