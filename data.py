"""
Looking at the dataset.
"""

import random
import math
from collections import Counter


def main():
  with open("names.txt", "r") as file:
    char_freq = {}

    # longest name in file
    for line in file:
      line = line.replace('\n', '')
      line = "#" + line + "$"
      left, right = 0, 1

      while right < (len(line)):
        curr_line = line[left:right + 1]
        if curr_line not in char_freq:
          char_freq[curr_line] = 0
        char_freq[curr_line] += 1

        left += 1
        right += 1

    probabilities = {}
    sums = {}
    for k in char_freq.keys():
      first_char = k[0]

      meow = [j for j in char_freq.keys() if j[0] == first_char]
      total = sum(char_freq[m] for m in meow)

      sums[first_char] = 0

      for m in meow:
        p_occur_second_char = char_freq[m] / total 
        probabilities[m] = p_occur_second_char

        sums[first_char] += p_occur_second_char

    names = []

    for _ in range(5):
      name = "#"
      for _ in range(5):
        vals = [[v, p] for v, p in probabilities.items() if v[0] == name[-1]]
        weights = [v[1] for v in vals]
        chars = [v[0] for v in vals]

        pred_seq = random.choices(chars, weights=weights, k=1)
        next_char = pred_seq[0][1]
        name = name + next_char
        if next_char == '$':
          break

      if name[-1] != '$':
        name = name + '$'

      names.append(name)

    logs = {}
    # calculating log of each probability
    for key, val in probabilities.items():
      logs[key] = math.log(val)

    s = 0.0
    total_occs = sum(f for f in char_freq.values())
    for key in probabilities.keys():
      print(f"For {key}. %: {probabilities[key]}, log: {logs[key]}")
      s += (logs[key] * char_freq[key])

    print(f"average: {-1 * (s / total_occs)}")

main()
