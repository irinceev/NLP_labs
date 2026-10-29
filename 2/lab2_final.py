from pathlib import Path

import gensim

MODEL_PATH = Path(__file__).parent / "cbow.txt"
POSITIVE = ["банка_NOUN", "кочан_NOUN", "квашение_NOUN", "крыша_NOUN", "консервы_NOUN", "пробка_NOUN", "крыша_NOUN", "люк_NOUN", "огурец_NOUN"]
NEGATIVE = ["бутыль_NOUN", "тушенка_NOUN", "дерево_NOUN", "кирпич_NOUN", "сушка_NOUN", "окно_NOUN", "корабль_NOUN", "провансаль_NOUN"]
TARGETS = {"крышка_NOUN", "капуста_NOUN"}
TOP_N = 10


def find_nouns(model, positive, negative, top_n):
    n = max(top_n, 20)

    while True:
        words = model.most_similar(
            positive=positive,
            negative=negative,
            topn=n
        )
        result = [
            (word, similarity)
            for word, similarity in words
            if word.endswith("_NOUN")
        ]

        if len(result) >= top_n:
            return result[:top_n]

        if n >= len(model):
            raise ValueError("Не удалось найти нужное количество существительных")

        n = min(n * 2, len(model))


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Файл модели не найден: {MODEL_PATH}")

    model = gensim.models.KeyedVectors.load_word2vec_format(
        str(MODEL_PATH),
        binary=False
    )

    absent = [word for word in POSITIVE + NEGATIVE if word not in model]
    if absent:
        raise ValueError(f"Отсутствуют слова: {', '.join(absent)}")

    similar = find_nouns(model, POSITIVE, NEGATIVE, TOP_N)
    positions = {word: i for i, (word, _) in enumerate(similar, 1)}

    def clean(words):
        return ", ".join(w.removesuffix("_NOUN") for w in words)

    print(f"Положительные: {clean(POSITIVE)}")
    print(f"Отрицательные: {clean(NEGATIVE)}")
    print()

    print(f"Ближайшие существительные (топ-{TOP_N}):")
    for i, (word, similarity) in enumerate(similar, 1):
        print(f"{i:>4}. {word.removesuffix('_NOUN'):<15} {similarity:.4f}")
    print()

    print("Целевые слова:")
    for word in sorted(TARGETS):
        rank = positions.get(word)
        result = f"позиция {rank}" if rank is not None else "нет в топе"
        print(f"  {word.removesuffix('_NOUN')}: {result}")

    if not TARGETS.issubset(positions):
        raise AssertionError("Не все целевые слова попали в топ-10")


if __name__ == "__main__":
    main()