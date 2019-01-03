from bs4 import BeautifulSoup as bs
import urllib.request
import re
import nltk
#nltk.download()
import contractions
import inflect

site = 'http://www.paulgraham.com/'

root_page = urllib.request.urlopen(site + 'articles.html')
root_soup = bs(root_page, 'html.parser')
#print(root_soup.prettify())
#tags = root_soup.findAll('font', face='veranda')
tags = root_soup.find_all('font', face='verdana')
tags = root_soup.find_all('a')
print()
print(tags[:-1])
print()
print(len(tags))

wordset = set()     # initialise empty set of words
bodies = list()      # initialise empty set to contain bodies of text gathered

with open('text.txt', 'w') as f:
    for tag_idx in range(1, 174):
        print(tags[tag_idx])
        href = tags[tag_idx].attrs['href']
        page = urllib.request.urlopen(site + href)
        page_soup = bs(page, 'html.parser')
        #print(page_soup.prettify())
        text = page_soup.find('font', face='verdana')
        #print(str(text))
        print('\n\n\n')
        text = text.text                        # get just the text (remove tags etc)
        f.write(text)
        text = re.sub('\[[^]]*\]', '', text)            # remove square brackets
        text = contractions.fix(text)                   # replaced contractions with their full words
        words = nltk.word_tokenize(text)                # make list of word tokens
        words = [word.lower() for word in words]        # lowercase
        words = [re.sub(r'[^\w\s]', '', word) for word in words]    # replace punctuation with empty string
        words = [word for word in words if word != '']  # remove empty strings
        bodies.append(words)
        print(bodies)
        wordset = wordset.union(words)      # add any new words in this body to the set of words
        break

print('Number of individual words:', len(wordset))

#print(text)
print(words)
