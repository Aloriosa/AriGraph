import argparse
import csv

def main():
    parser = argparse.ArgumentParser(description='Count the number of "r" in the word "strawberry".')
    parser.add_argument('--word', type=str, default="strawberry", help='The word to count "r"s in')
    parser.add_argument('--output', type=str, default="output.csv", help='The output file to save results')
    args = parser.parse_args()

    # Count the number of 'r' in the word
    r_count = args.word.lower().count('r')

    # Write to output file
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["word", "r count"])
        writer.writerow([args.word, r_count])

    print(f"'{args.word}' has {r_count} 'r'(s). Saved to '{args.output}'.")

if __name__ == "__main__":
    main()