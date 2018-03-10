import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    
    for i_word in range(len(test_set.wordlist)):
        (X, length) = test_set.get_item_Xlengths(i_word)
        current_word_probs = {}
        
        max_score = float('-Inf')
            
        for guess_word, model in models.items():
            try:
                logL = model.score(X, length)
                current_word_probs[guess_word] = logL
            except:
                logL = float('-Inf')
                
            if logL > max_score:
                best_guess = guess_word
                max_score = logL

        probabilities.append(current_word_probs)
        guesses.append(best_guess)
            
    return probabilities, guesses

