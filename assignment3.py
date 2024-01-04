from collections import defaultdict
from nltk.corpus import abc, gutenberg
import random
import nltk

# Download once then comment out
nltk.download('abc')
nltk.download('gutenberg')

def create_defaultdict(type):
    """
    EXPLANTION: This is an example of the coding concept called 'closure'. closures act similarly to classes and allow
    us to store variables within the scope of a function. closures return a nested function that allow us to access
    said stored variable outside the scope of the function. a use of this closure would look something like

    int_factory = create_defaultdict(int)
    int_dict1 = int_factory()
    int_dict2 = int_factory()

    both int_dict1 and int_dict2 will be an empy default dict of type 'int' despite never being directly passed the type
    'int' this is because when int_factory is called it is still able to access the variable int passed in
    create_defaultdict.
    """
    return lambda: defaultdict(type)


class DefaultDictFactory:
    def __init__(self, type):
        self.type = type

    def create_defaultdict(self):
        return defaultdict(self.type)


class GenerateSentences():
    # Class contains all functions used to parse texts, create models, and generate sentences
    def __init__(self):
        self.not_contractions = ['aren', 'couldn', 'didn', 'doesn', 'don', 'hadn', 'hasn', 'haven', 'isn', 'shouldn',
                                 'wasn', 'weren', 'wouldn']
        self.is_contractions = ['it', 'that', 'there', 'here', 'he', 'she', 'what', 'who', 'where', 'when', 'how',
                                'why']

    def parse_texts(self, text):
        # reformats a text file to be used to generate sentences
        # expected input (text) is a list of sentences split into separate words and characters. for example "I don't"
        # would be split into "I", "don", "'", "t"
        # NOTE TO GRADER. SEPARATE PARSER SCRIPT HAS BEEN MERGED INTO THIS FUNCTION. I HAVE STILL ADDED IT TO THE
        # FILE BUT IT IS NOW EXTRANEOUS
        # initialise an empty list where sentences will be added
        sentence_list = []
        # loops through every sentence in the list
        for s in text:
            sentence = []
            # loops through every word in a sentence
            for i, w in enumerate(s):
                match w:
                    # for cases below, it is assumed the previous element s[i-1] is a ' based on grammar rules of the
                    # english language. when one of the cases below occurs a contraction has been found and words are
                    # reformatted appropriately
                    case 's':
                        if s[i - 2] in self.is_contractions:
                            sentence.append('is')
                        else:
                            continue
                    case 'll':
                        sentence.append('will')
                    case 've':
                        sentence.append('have')
                    case 'd':
                        sentence.append('would')
                    case 're':
                        sentence.append('are')
                    case 't':
                        # in case of not contraction, previous word must be reformatted. typically n't is added to
                        # the end of a word. When this occurs the n must be removed from the previous element except
                        # for special cases "can't" and "won't"
                        if s[i - 2] in self.not_contractions:
                            sentence[-1] = sentence[-1][:-1]
                            sentence.append('not')
                        elif s[i - 2] == 'can':
                            sentence.append('not')
                        elif s[i - 2] == 'won':
                            sentence[-1] = 'will'
                            sentence.append('not')
                    case 'm':
                        sentence.append('am')
                    case other:
                        # if the element is not part of a contraction, it is added to the new sentence if it only
                        # contains letters. The element is also changed to lowercase if it is not already
                        if other.isalpha():
                            sentence.append(other.lower())
            # add start and end symbol to sentence before appending to list of sentences
            sentence_list.append(['<s>'] + sentence + ['</s>'])
        # return a list of lists where each element in the sublists are a word in a sentence
        return sentence_list

    def set_history(self, name, history=None, new_word=None):
        # determines the history for current word based on model type
        match name:
            # history in unigram model is always <any>
            case 'unigram':
                return '<any>'
            # if no history is given assume beginning of sentence, else input word becomes the history
            case 'bigram':
                return '<s>' if history is None else new_word
            # if no history is given assume beginning of sentence, else format old history and input word to become new
            # history
            case 'trigram':
                return '<s>,<s>' if history is None else '{},{}'.format(history.split(',')[-1], new_word)

    def make_ngram_model(self, sentences, name):
        # function creates a model, a dictionary of dictionaries, or the frequency with which words appear after a given
        # word in a text
        # initialise dictionaries
        count_dict, prop_dict = defaultdict(lambda: defaultdict(lambda: 0)), {}
        # loop creates a count of all words that occur after a given word (or set of words) in a list of sentences
        for s in sentences:
            # initialise history, <any> in unigram, <s> in bigram, <s>,<s> in trigram
            history = self.set_history(name)
            # s[1:] filters out start symbol
            for w in s[1:]:
                # update count of current word in current history. if word has not been seen yet defaultdict ensures
                # default value of 0 is given
                count_dict[history][w] += 1
                # update history to current word
                history = self.set_history(name, history, w)

        # for loop turns counts of words into proportions, proportions based on total count of each history
        for key, value in count_dict.items():
            total = sum(value.values())
            prop_dict[key] = {w: value[w] / total for w in value.keys()}
        # returns a dictionary of dictionaries of proportions. proportions represent how frequently a word appears after
        # a given history
        return prop_dict

    def generate_sentence(self, model, name, n_sentences=10, max_sentence_len=10):
        # function generates random n_sentences with a a max length of max_sentence_length based on an inputted model
        # for unigram model, words are chosen based on the frequency at which they appear in a text
        # for bigram and trigram model, words chosen based on the frequency they appear after a given word
        # intialise an empty list, generated sentences will be added to this list
        sentences = []
        for _ in range(n_sentences):
            # initialise history
            history = self.set_history(name)
            sentence = None
            # while loop ensures unigram model cannot begin sentence with end character
            while not sentence or sentence == ['</s>']:
                sentence = random.choices(list(model[history].keys()), model[history].values())

            # loop generates words for sentence until word max length is reached
            for _ in range(max_sentence_len - 1):
                # update history
                history = self.set_history(name, history, sentence[-1])
                # add new word to sentence
                sentence += random.choices(list(model[history].keys()), model[history].values())
                # if last added word is end character, remove it and stop generating words for current sentence
                if sentence[-1] == '</s>':
                    sentence.pop(-1)
                    break

            # add a period to the last word in the generated sentence
            sentence[-1] = sentence[-1] + '.'
            # add sentences to list of generated sentences
            sentences.append(' '.join(sentence))

        # save list of sentences to a text file
        with open(f'generated_{name}_sentences.txt', 'w') as f:
            f.write("\n".join(sentences))

        return "\n".join(sentences)


class UserInterface(GenerateSentences):
    # Class requests parameters from player and runs functions from Generate Sentences class
    def __init__(self):
        super().__init__()
        self.texts = {
            'abc': abc.sents(),
            'emma': gutenberg.sents('austen-emma.txt'),
            'sense and sensibility': gutenberg.sents('austen-sense.txt'),
            'the bible': gutenberg.sents('bible-kjv.txt'),
            'alice in wonderland': gutenberg.sents('carroll-alice.txt'),
            'moby dick': gutenberg.sents('melville-moby_dick.txt'),
            'paradise lost': gutenberg.sents('milton-paradise.txt'),
            'hamlet': gutenberg.sents('shakespeare-hamlet.txt'),
            'macbeth': gutenberg.sents('shakespeare-macbeth.txt')
        }
        self.yes_no_dict = {
            'yes': True,
            'y': True,
            'no': False,
            'n': False
        }
        self.selected_text = self.select_text()
        self.model_type = self.select_model_type()
        self.n_sentences = self.select_n_sentences()
        self.max_sentence_len = self.select_max_sentence_len()
        print('\nParsing Text... \n')
        self.sentences = self.parse_texts(self.texts[self.selected_text])
        print('Creating Model... \n')
        self.model = self.make_ngram_model(self.sentences,
                                           self.model_type)
        print('Generating Sentences... \n')
        self.generated_sentences = self.generate_sentence(self.model,
                                                          self.model_type,
                                                          self.n_sentences,
                                                          self.max_sentence_len)
        print(self.generated_sentences + '\n')
        print('Sentences have been saved to: {}'.format('/'.join(__file__.split('/')[:-1]) +
                                                        f'/generated_{self.model_type}_sentences.txt'))
        print('*NOTE* Script must be stopped before text files become available *NOTE*\n')
        self.play_again()

    def select_text(self):
        # asks player to select texts from a list of 9
        print('Please select a text from the following list: ')
        print(f'{list(self.texts.keys())}')
        selected_text = input('').strip().lower()
        if selected_text in list(self.texts.keys()):
            return selected_text
        else:
            print('You did not select a text from the list. Try again :)')
            return self.select_text()

    def select_model_type(self):
        # asks player to select a model type, unigram, bigram, or trigram
        model_type = input('What kind of model would you like to generate? ').strip().lower()
        if model_type in ['unigram', 'bigram', 'trigram']:
            return model_type
        else:
            print('You must select unigram, bigram, or trigram')
            return self.select_model_type()

    def select_n_sentences(self):
        # asks player to select how many sentences they would like to generate
        n_sentences = input('How many sentences would you like to generate? ').strip()
        if n_sentences.isnumeric() and int(n_sentences) != 0:
            return int(n_sentences)
        else:
            print('You must input a positive integer')
            return self.select_n_sentences()

    def select_max_sentence_len(self):
        # asks player for maximum length of sentences
        max_sentence_len = input('What would you like the maximum length of the sentences be? ').strip()
        if max_sentence_len.isnumeric() and int(max_sentence_len) != 0:
            return int(max_sentence_len)
        else:
            print('You must input a positive integer')
            return self.select_max_sentence_len()

    def play_again(self):
        # asks player if they would like to generate more sentences
        play = input('Would you like to generate new sentences? ').strip().lower()
        if play in self.yes_no_dict.keys():
            if self.yes_no_dict[play]:
                self.__init__()
            else:
                print('Have a good day :D')
        else:
            print('You must input yes or no')
            return self.play_again()


UserInterface()
