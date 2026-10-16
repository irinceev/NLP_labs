import sys
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import pymorphy3

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)  # нужно в новых версиях NLTK

morph = pymorphy3.MorphAnalyzer()

# Части речи: существительные и полные прилагательные
POS_OK = {'NOUN', 'ADJF'}

# «Вторые» падежи pymorphy3 приводим к основным (в лесу = loc2 -> loct и т.п.)
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
    result = []
    for sentence in sent_tokenize(text, language='russian'):
        tokens = word_tokenize(sentence, language='russian')
        parsed = [options(t) if is_word(t) else [] for t in tokens]

        for left, right in zip(parsed, parsed[1:]):
            # pymorphy3 не видит контекст, поэтому проверяем все разборы
            # и берём согласованную пару с наибольшей вероятностью
            pairs = [(a, b) for a in left for b in right if agree(a, b)]
            if pairs:
                a, b = max(pairs, key=lambda p: p[0].score * p[1].score)
                result.append((a.normal_form, b.normal_form))
    return result


def main():
    with open(sys.argv[1], encoding='utf-8') as f:
        text = f.read()

    pairs = find_pairs(text)
    for first, second in pairs:
        print(first, second)
    print(f'\nВсего пар: {len(pairs)}')


if __name__ == '__main__':
    main()