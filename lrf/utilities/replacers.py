import re, csv, yaml, enchant
from nltk.corpus import wordnet
from nltk.metrics import edit_distance
from lrf.configs import lrf_config

replacement_patterns = [
    (r'won\'t','will not'),
    (r'can\'t', 'cannot'),
	(r'i\'m', 'i am'),
	(r'ain\'t', 'is not'),
	(r'(\w+)\'ll', '\g<1> will'),
	(r'(\w+)n\'t', '\g<1> not'),
	(r'(\w+)\'ve', '\g<1> have'),
	(r'(\w+)\'s', '\g<1> is'),
	(r'(\w+)\'re', '\g<1> are'),
    (r'(\w+)\'d', '\g<1> would')
]

intensifiers,diminishers = lrf_config.get_intensifier_and_diminishers()

class RegexpReplacer(object):
    """ Replaces regular expression in a text.
    'cannot is a contraction'
    'I should have done that thing I did not do'
    """

    def __init__(self, patterns=replacement_patterns):
        self.patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]

    def replace(self, text):
        s = text

        for (pattern, repl) in self.patterns:
            s = re.sub(pattern, repl, s)

        return s


####################################
## Replacing Repeating Characters ##
####################################

class RepeatReplacer(object):
    """ Removes repeating characters until a valid word is found.
    'love'
    'ooh'
    'goose'
    """

    def __init__(self):
        self.repeat_regexp = re.compile(r'(\w*)(\w)\2(\w*)')
        self.repl = r'\1\2\3'
    def intensifier_and_spell_chck(self,word):
        is_intensifier = 0
        if word.isupper():
            is_intensifier = 1

        count = 0
        res, cnt = self.replace(word, count)
        if wordnet.synsets(word) == [] and count == 0:
            res = SpellingReplacer().replace(word)

        is_intensifier = 1 if is_intensifier==1 or cnt!=0 or res in intensifiers else 0
        is_diminisher = 1 if res in diminishers else 0

        return res,is_intensifier,is_diminisher

    def replace(self, word,count):
        if wordnet.synsets(word):
            return word,count

        repl_word = self.repeat_regexp.sub(self.repl, word)

        if repl_word != word:

            return self.replace(repl_word,count),count

        else:

            return repl_word,count


######################################
## Spelling Correction with Enchant ##
######################################

class SpellingReplacer(object):
    """ Replaces misspelled words with a likely suggestion based on shortest
    edit distance.
    'cookbook'
    """

    def __init__(self, dict_name='en', max_dist=3):
        self.spell_dict = enchant.Dict(dict_name)
        self.max_dist = max_dist

    def replace(self, word):
        if self.spell_dict.check(word):
            return word

        suggestions = self.spell_dict.suggest(word)

        if suggestions and edit_distance(word, suggestions[0]) <= self.max_dist:
            return suggestions[0]
        else:
            return word


class CustomSpellingReplacer(SpellingReplacer):
    """ SpellingReplacer that allows passing a custom enchant dictionary, such
    a DictWithPWL.
    'nltk'
    """

    def __init__(self, spell_dict, max_dist=2):
        self.spell_dict = spell_dict
        self.max_dist = max_dist


########################
## Replacing Synonyms ##
########################

class WordReplacer(object):
    """ WordReplacer that replaces a given word with a word from the word_map,
    or if the word isn't found, returns the word as is.
    'birthday'
    'happy'
    """

    def __init__(self, word_map):
        self.word_map = word_map

    def replace(self, word):
        return self.word_map.get(word, word)


class CsvWordReplacer(WordReplacer):
    """ WordReplacer that reads word mappings from a csv file.
    'birthday'
    'happy'
    """

    def __init__(self, fname):
        word_map = {}

        for line in csv.reader(open(fname)):
            word, syn = line
            word_map[word] = syn

        super(CsvWordReplacer, self).__init__(word_map)


class YamlWordReplacer(WordReplacer):
    """ WordReplacer that reads word mappings from a yaml file.
    'birthday'
    'happy'
    """

    def __init__(self, fname):
        word_map = yaml.load(open(fname))
        super(YamlWordReplacer, self).__init__(word_map)


#######################################
## Replacing Negations with Antonyms ##
#######################################

class AntonymReplacer(object):
    def replace(self, word, pos=None):
        """ Returns the antonym of a word, but only if there is no ambiguity.
        'beautify'
        'uglify'
        """
        antonyms = set()

        for syn in wordnet.synsets(word, pos=pos):
            for lemma in syn.lemmas():
                for antonym in lemma.antonyms():
                    antonyms.add(antonym.name())

        if len(antonyms) == 1:
            return antonyms.pop()
        else:
            return None

    def replace_negations(self, sent):
        """ Try to replace negations with antonyms in the tokenized sentence.
        ['do', 'beautify', 'our', 'code']
        ['good', 'is', 'not', 'evil']
        """
        i, l = 0, len(sent)
        words = []

        while i < l and sent[i] not in [',',':',';','!','?','[',']','.','but']:
            word = sent[i]

            if word == 'not' and i + 1 < l:
                ant = self.replace(sent[i + 1])

                if ant:
                    words.append(ant)
                    i += 2
                    continue
                words.append(word)
            if word == 'neither' and i + 1 < l:
                neither_wrd = sent[i+1]
                neither_wrd_ant = self.replace(sent[i+1])
                i += 2
                continue
            if word == 'nor' and neither_wrd_ant is not None and i+1 < l:
                nor_wrd = sent[i+1]
                nor_wrd_ant = self.replace(sent[i+1])
                if nor_wrd_ant:
                    words.append(neither_wrd_ant)
                    words.append('and')
                    words.append(nor_wrd_ant)
                    i += 2
                else:
                    words.append('neither')
                    words.append(neither_wrd)
                    words.append('nor')
                    words.append(nor_wrd)
                    i += 2
            else:
                words.append(word)
                i += 1

        words = words + sent[i:]

        return words


class AntonymWordReplacer(WordReplacer, AntonymReplacer):
    """ AntonymReplacer that uses a custom mapping instead of WordNet.
    Order of inheritance is very important, this class would not work if
    AntonymReplacer comes before WordReplacer.
    ['good', 'is', 'good']
    """
    pass


if __name__ == '__main__':
    # import doctest
    # doctest.testmod()
    print('HELLO')