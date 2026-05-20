import os
import argparse
from datasets import load_dataset
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pg19', help='Dataset to use (pg19, fanfics, random)')
    parser.add_argument('--output', type=str, default='data/pg19_sample.txt', help='Output file')
    parser.add_argument('--sample_size', type=int, default=100, help='Sample size')
    args = parser.parse_args()

    os.makedirs('data', exist_ok=True)

    if args.dataset == 'pg19':
        print("Loading PG-19 dataset...")
        dataset = load_dataset('pg19', split='train')
        texts = []
        for item in dataset:
            if len(item['text']) > 1000:  # Filter very short texts
                texts.append(item['text'])
        print(f"Loaded {len(texts)} texts")
        sample = random.sample(texts, min(args.sample_size, len(texts)))
        with open(args.output, 'w') as f:
            for text in sample:
                f.write(text + '\n')
        print(f"Saved {len(sample)} samples to {args.output}")

    elif args.dataset == 'fanfics':
        print("Generating synthetic fanfic-like texts...")
        # Simulate fanfic texts (since we can't download AO3 in Docker)
        # In practice, you'd download from AO3 API or use pre-downloaded data
        from faker import Faker
        fake = Faker()
        texts = []
        for _ in range(args.sample_size):
            title = fake.sentence(nb_words=6)
        texts.append(title + '\n')
        for _ in range(random.randint(5, 10)):
            paragraph = fake.paragraph(nb_sentences=4)
        texts.append(paragraph + '\n')
        texts.append('\n' + '='*80 + '\n\n')
        with open(args.output, 'w') as f:
            f.writelines(texts)
        print(f"Generated {len(texts)} synthetic fanfic texts")

    elif args.dataset == 'random':
        print("Generating random word sequences...")
        import nltk
        try:
            nltk.data.find('corpora/words')
        except LookupError:
            nltk.download('words')
        from nltk.corpus import words
        word_list = [w.lower() for w in words.words() if len(w) > 3 and w.isalpha()]
        texts = []
        for _ in range(args.sample_size):
            length = random.randint(10, 20)
        sentence = ' '.join(random.choices(word_list, k=length))
        texts.append(sentence + '\n')
        with open(args.output, 'w') as f:
            f.writelines(texts)
        print(f"Generated {len(texts)} random sequences")

if __name__ == "__main__":
    main()