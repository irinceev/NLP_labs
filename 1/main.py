import sys
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import pymorphy3

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True) 

morph = pymorphy3.MorphAnalyzer()

#существительные и полные прилагательные
POS_OK = {'NOUN', 'ADJF'}

CASES = {'gen1': 'gent', 'gen2': 'gent',
         'acc2': 'accs',
         'loc1': 'loct', 'loc2': 'loct',
         'voct': 'nomn'}

def is_word(token):
    return any(ch.isalpha() for ch in token)

def options(token: str):
    parses = morph.parse(token)
    if parses[0].tag.POS not in POS_OK:
        return []
    return [p for p in parses if p.tag.POS in POS_OK]

def case(p):
    return CASES.get(p.tag.case, p.tag.case)

def agree(a, b):
    if a.tag.number != b.tag.number or case(a) != case(b):
        return False
    return a.tag.number == 'plur' or a.tag.gender == b.tag.gender

def find_pairs(text: str):
    """Возвращает список словарей: исходная пара слов и её нормальная форма."""
    result = []
    for sentence in sent_tokenize(text, language='russian'):
        tokens = word_tokenize(sentence, language='russian')
        parsed = [options(t) if is_word(t) else [] for t in tokens]

        for i in range(len(tokens) - 1):
            left, right = parsed[i], parsed[i + 1]

            pairs = [(a, b) for a in left for b in right if agree(a, b)]
            if pairs:
                a, b = max(pairs, key=lambda p: p[0].score * p[1].score)
                result.append({
                    'orig': (tokens[i], tokens[i + 1]),
                    'norm': (a.normal_form, b.normal_form),
                })
    return result

def main():
    with open(sys.argv[1], encoding='utf-8') as f:
        text = f.read()

    pairs = find_pairs(text)
    for p in pairs:
        orig = f'{p["orig"][0]} {p["orig"][1]}'
        norm = f'{p["norm"][0]} {p["norm"][1]}'
        print(f'{orig:<35} →  {norm}')
    print(f'\nВсего пар: {len(pairs)}')


if __name__ == '__main__':
    main()
