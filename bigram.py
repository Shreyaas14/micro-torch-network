"""
Bi-gram model which predicts the next character 
based on character pair frequencies.
Uses PyTorch.
"""

import argparse
from pathlib import Path

import torch 
import torch.nn.functional as F

## Consts
LEARNING_RATE = 50
ITERATIONS = 1000
START_CHAR = "#"
END_CHAR = "$"

## Methods for extracting and formatting the names.
def process_line(line: str) -> str:
  line = line.replace('\n', '')
  line = "#" + line + '$' 

  if line == "":
      return ""

  return line

def file_contents(filename: str, permissions: str):
  with open(filename, permissions) as file: 
    names = [process_line(line) for line in file]

  return names 

def get_tokens(names: list[str]):
  letters = set()

  for name in names:
    chars = list(name)
    for char in chars:
      if char not in letters:
        letters.add(char)

  return sorted(letters)

# building string to int dict
def build_stoi(tokens: list[str]):
  stoi = {}

  for i in range(len(tokens)):
    stoi[tokens[i]] = i

  return stoi

# building int to string dict
def build_itos(tokens: list[str]):
  itos = {}

  for i in range(len(tokens)):
    itos[i] = tokens[i]

  return itos

def names_to_tensors(names: list[str], 
                     stoi: dict[str, int]
                     ) -> tuple[torch.Tensor, torch.Tensor]:
  xs, ys = [], []
  for name in names: 
    left, right = 0, 1 

    while right < (len(name)):
      curr_line = name[left:right + 1]
      xs.append(stoi[curr_line[0]])
      ys.append(stoi[curr_line[1]])

      left += 1; right += 1

  xs = torch.tensor(xs)
  ys = torch.tensor(ys)

  return (xs, ys)


def calculate_loss(pred: torch.Tensor) -> torch.Tensor:
  logs = -torch.log(pred)
  nll = torch.sum(logs) / len(logs)

  return nll

@torch.no_grad()
def generate_name(weights, stoi, itos):
  current_char = stoi[START_CHAR]
  name = START_CHAR

  while name[-1] != END_CHAR:
    o_h = F.one_hot(torch.tensor(current_char), num_classes=len(stoi))
    char_dist = F.softmax(o_h.float() @ weights, dim=0)
    current_char = torch.multinomial(char_dist, num_samples=1).item()
    next_char = itos[current_char]

    name += next_char

  return name
  

def main():
  parser = argparse.ArgumentParser(description="Train a bigram model or generate names from saved weights.")
  parser.add_argument("--use-weights", action="store_true",
                      help="Load weights.pt and generate names without training.")
  args = parser.parse_args()

  data_dir = Path(__file__).resolve().parent
  weights_path = data_dir / "weights.pt"
  raw_names = file_contents(data_dir / "names.txt", "r")
  tokens = get_tokens(raw_names)
  stoi, itos = build_stoi(tokens), build_itos(tokens)

  if args.use_weights:
    if not weights_path.is_file():
      parser.error(f"Saved weights not found: {weights_path}. Run without --use-weights to train first.")
    W = torch.load(weights_path, map_location="cpu", weights_only=True)
  else:
    ## tensors for the training data
    xs, ys = names_to_tensors(raw_names, stoi)

    if xs is None or ys is None:
      print("Error occurred while creating training data.")

    # one-hot encoded inputs (xenc means x-encoded)
    xenc = F.one_hot(xs, num_classes=len(stoi))

    # weights
    W = torch.randn((len(stoi), len(stoi)), requires_grad=True)

    for i in range(ITERATIONS):
      ## FORWARD PASS
      # logits (raw scores). 
      # one_hot returns ints, so we use float() to convert them to floats
      logits = xenc.float() @ W 

      # puts all values thru e^x; values stay positive & larger values are larger after e^x
      counts = logits.exp()

      # sum of each row (belief of next character)
      c_sum = torch.sum(counts, dim=1, keepdim=True)

      # probabilities
      probs = counts / c_sum

      outputs = probs[torch.arange(0, len(ys)), ys]

      ## BACKWARD PASS
      nll = calculate_loss(outputs)
      nll.backward()
      print(f"Iteration {i} loss: {nll}.")

      if W.grad is None:
        print("no gradient, can't do backward pass.")
      else:
        with torch.no_grad():
          W.sub_(W.grad.mul_(LEARNING_RATE)) 
          W.grad = None

    torch.save(W, weights_path)

  names = []
  for i in range(5):
    curr_name = generate_name(W, stoi, itos)
    names.append(curr_name)

  print(names)

if __name__ == "__main__":
  main()
  
