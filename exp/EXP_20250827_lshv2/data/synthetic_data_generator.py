"""
Synthetic Data Generator for LSH v2 Training
============================================

Generates synthetic training data for autoregressive sequence modeling:
1. Set of Data tasks: Data processing with hash functions
2. Odd One Out tasks: Pattern recognition and anomaly detection
3. Fuzzy Regex Parsing: Pattern matching with varying complexity

Designed for training models on computational reasoning without language understanding.

Author: Research Implementation
Version: 2.0
"""

import numpy as np
import random
import re
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class SyntheticTask:
    """Configuration for a synthetic task"""

    name: str
    description: str
    input_format: str
    output_format: str
    difficulty_levels: List[str]


class SyntheticDataGenerator:
    """Generator for synthetic training data with long sequences"""

    def __init__(self, vocab_size: int = 32000, max_seq_len: int = 131072):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Special tokens
        self.pad_token = 0
        self.bos_token = 1
        self.eos_token = 2
        self.sep_token = 3

        # Task tokens
        self.task_tokens = {
            "set_data": 10,
            "odd_one_out": 11,
            "fuzzy_regex": 12,
            "hash_func": 20,
            "output": 21,
            "items": 30,
            "pattern": 31,
        }

        # Create vocabulary mapping
        self.vocab = self._create_vocab()
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

        # Task definitions
        self.tasks = {
            "set_data": SyntheticTask(
                name="Set of Data",
                description="Apply hash functions to structured data",
                input_format="[SET_DATA] items [HASH_FUNC] function [OUTPUT] result",
                output_format="Bucketed data based on hash function",
                difficulty_levels=["basic", "medium", "hard", "extreme"],
            ),
            "odd_one_out": SyntheticTask(
                name="Odd One Out",
                description="Identify items that don't belong in patterns",
                input_format="[ITEMS] item1, item2, ..., itemN [OUTPUT] odd_item",
                output_format="Item that breaks the pattern",
                difficulty_levels=["basic", "medium", "hard", "extreme"],
            ),
            "fuzzy_regex": SyntheticTask(
                name="Fuzzy Regex Parsing",
                description="Parse and match patterns with fuzzy logic",
                input_format="[PATTERN] regex [TEXT] input [OUTPUT] matches",
                output_format="Matched patterns from input text",
                difficulty_levels=["basic", "medium", "hard", "extreme"],
            ),
        }

    def _create_vocab(self) -> Dict[str, int]:
        """Create vocabulary mapping"""
        vocab = {
            "<PAD>": self.pad_token,
            "<BOS>": self.bos_token,
            "<EOS>": self.eos_token,
            "<SEP>": self.sep_token,
        }

        # Add task tokens
        for name, token_id in self.task_tokens.items():
            vocab[f"<{name.upper()}>"] = token_id

        # Add numbers (0-9999)
        for i in range(10000):
            vocab[str(i)] = 100 + i

        # Add letters and symbols
        for i in range(26):
            vocab[chr(ord("a") + i)] = 15000 + i
            vocab[chr(ord("A") + i)] = 15100 + i

        for symbol in ".,;:!?()[]{}+-*/=<>@#$%^&|~`\"'\\":
            vocab[symbol] = 15200 + ord(symbol)

        # Fill remaining vocabulary
        current_id = 20000
        while current_id < self.vocab_size:
            token = f"tok_{current_id}"
            vocab[token] = current_id
            current_id += 1

        return vocab

    def tokenize(self, text: str) -> List[int]:
        """Simple tokenization"""
        tokens = []
        words = text.split()

        for word in words:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                # Character-level fallback
                for char in word:
                    if char in self.vocab:
                        tokens.append(self.vocab[char])
                    else:
                        tokens.append(
                            self.vocab.get(
                                f"tok_{hash(char) % 1000 + 20000}", self.pad_token
                            )
                        )

        return tokens

    def detokenize(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text"""
        tokens = []
        for token_id in token_ids:
            if token_id in self.reverse_vocab:
                tokens.append(self.reverse_vocab[token_id])
            else:
                tokens.append(f"<UNK_{token_id}>")

        return " ".join(tokens)

    def generate_set_data_task(
        self, difficulty: str = "medium", target_length: int = 8192
    ) -> Dict[str, Any]:
        """Generate Set of Data task for autoregressive sequence modeling"""

        if difficulty == "basic":
            num_items = random.randint(20, 100)
            num_hash_operations = random.randint(2, 5)
        elif difficulty == "medium":
            num_items = random.randint(100, 500)
            num_hash_operations = random.randint(5, 12)
        elif difficulty == "hard":
            num_items = random.randint(500, 2000)
            num_hash_operations = random.randint(12, 25)
        else:  # extreme
            num_items = random.randint(2000, 8000)
            num_hash_operations = random.randint(25, 50)

        # Generate raw data items
        items = []
        data_type = random.choice(["integers", "floats", "strings", "tuples", "mixed"])

        if data_type == "integers":
            for _ in range(num_items):
                items.append(str(random.randint(1, 99999)))
        elif data_type == "floats":
            for _ in range(num_items):
                value = round(random.uniform(0.1, 999.9), 2)
                items.append(f"{value:.2f}")
        elif data_type == "strings":
            for _ in range(num_items):
                length = random.randint(3, 12)
                chars = "0123456789abcdef"
                string = "".join(random.choices(chars, k=length))
                items.append(string)
        elif data_type == "tuples":
            for _ in range(num_items):
                tuple_size = random.randint(2, 4)
                values = [str(random.randint(0, 999)) for _ in range(tuple_size)]
                items.append("(" + ",".join(values) + ")")
        else:  # mixed
            for _ in range(num_items):
                item_type = random.choice(["int", "float", "string", "tuple"])
                if item_type == "int":
                    items.append(str(random.randint(1, 99999)))
                elif item_type == "float":
                    value = round(random.uniform(0.1, 999.9), 2)
                    items.append(f"{value:.2f}")
                elif item_type == "string":
                    length = random.randint(4, 8)
                    chars = "0123456789abcdef"
                    string = "".join(random.choices(chars, k=length))
                    items.append(string)
                else:  # tuple
                    tuple_size = random.randint(2, 3)
                    values = [str(random.randint(0, 99)) for _ in range(tuple_size)]
                    items.append("(" + ",".join(values) + ")")

        # Build sequence: DATA -> HASH_FUNC -> OUTPUT
        text_parts = ["<SET_DATA>"]
        text_parts.extend(items)

        # Apply hash/bucketing functions
        for hash_op in range(num_hash_operations):
            hash_function = random.choice(
                [
                    "modulo",
                    "sum_digits",
                    "first_char",
                    "length_mod",
                    "checksum",
                    "xor_hash",
                    "bucket_sort",
                    "range_partition",
                ]
            )

            text_parts.append("<HASH_FUNC>")

            if hash_function == "modulo":
                mod_value = random.choice([3, 4, 5, 7, 8, 11])
                text_parts.append(f"MOD_{mod_value}")

                buckets = {i: [] for i in range(mod_value)}
                for item in items:
                    try:
                        if "." in item:
                            value = int(float(item))
                        elif item.isdigit():
                            value = int(item)
                        elif item.startswith("("):
                            nums = item.strip("()").split(",")
                            value = int(nums[0])
                        else:
                            value = sum(ord(c) for c in item)

                        bucket = value % mod_value
                        buckets[bucket].append(item)
                    except Exception:
                        buckets[0].append(item)

                text_parts.append("<OUTPUT>")
                for bucket_id in range(mod_value):
                    if buckets[bucket_id]:
                        text_parts.append(f"B{bucket_id}:")
                        text_parts.extend(buckets[bucket_id][:20])

            elif hash_function == "sum_digits":
                mod_value = random.choice([3, 4, 5, 7])
                text_parts.append(f"SUM_DIGITS_MOD_{mod_value}")

                buckets = {i: [] for i in range(mod_value)}
                for item in items:
                    try:
                        digit_sum = sum(int(c) for c in item if c.isdigit())
                        bucket = digit_sum % mod_value
                        buckets[bucket].append(item)
                    except Exception:
                        buckets[0].append(item)

                text_parts.append("<OUTPUT>")
                for bucket_id in range(mod_value):
                    if buckets[bucket_id]:
                        text_parts.append(f"B{bucket_id}:")
                        text_parts.extend(buckets[bucket_id][:20])

            # Add other hash functions...
            elif hash_function == "first_char":
                text_parts.append("FIRST_CHAR_HASH")

                char_buckets = {}
                for item in items:
                    first_char = item[0] if item else "0"
                    if first_char not in char_buckets:
                        char_buckets[first_char] = []
                    char_buckets[first_char].append(item)

                text_parts.append("<OUTPUT>")
                for char in sorted(char_buckets.keys())[:10]:
                    text_parts.append(f"C{char}:")
                    text_parts.extend(char_buckets[char][:15])

        text_parts.append("<EOS>")
        full_text = " ".join(text_parts)

        # Tokenize
        tokens = self.tokenize(full_text)

        # Pad or truncate to target length
        if len(tokens) < target_length:
            tokens.extend([self.pad_token] * (target_length - len(tokens)))
        else:
            tokens = tokens[:target_length]

        return {
            "task": "set_data",
            "difficulty": difficulty,
            "text": full_text,
            "tokens": tokens,
            "num_items": num_items,
            "num_hash_operations": num_hash_operations,
            "data_type": data_type,
            "length": len(tokens),
        }

    def generate_odd_one_out_task(
        self, difficulty: str = "medium", target_length: int = 8192
    ) -> Dict[str, Any]:
        """Generate Odd One Out task"""

        if difficulty == "basic":
            num_sets = random.randint(5, 20)
            set_size = random.randint(4, 8)
        elif difficulty == "medium":
            num_sets = random.randint(20, 100)
            set_size = random.randint(8, 15)
        elif difficulty == "hard":
            num_sets = random.randint(100, 500)
            set_size = random.randint(15, 25)
        else:  # extreme
            num_sets = random.randint(500, 2000)
            set_size = random.randint(25, 50)

        text_parts = ["<ODD_ONE_OUT>"]

        for set_idx in range(num_sets):
            # Generate pattern with one odd item
            pattern_type = random.choice(["numbers", "sequences", "strings"])

            if pattern_type == "numbers":
                # Arithmetic sequence with one outlier
                base = random.randint(1, 100)
                step = random.randint(1, 10)
                pattern_items = [str(base + i * step) for i in range(set_size - 1)]

                # Add odd one out
                odd_item = str(random.randint(1, 10000))
                while odd_item in pattern_items:
                    odd_item = str(random.randint(1, 10000))

                items = pattern_items + [odd_item]
                random.shuffle(items)

            elif pattern_type == "sequences":
                # Geometric or other sequence
                base = random.randint(2, 10)
                pattern_items = [str(base**i) for i in range(1, set_size)]

                odd_item = str(random.randint(1, 1000))
                while odd_item in pattern_items:
                    odd_item = str(random.randint(1, 1000))

                items = pattern_items + [odd_item]
                random.shuffle(items)

            else:  # strings
                # String pattern (same length, prefix, etc.)
                target_len = random.randint(3, 8)
                pattern_items = []
                for _ in range(set_size - 1):
                    word = "".join(
                        random.choices("abcdefghijklmnopqrstuvwxyz", k=target_len)
                    )
                    pattern_items.append(word)

                # Different length odd item
                odd_len = target_len + random.choice([-2, -1, 1, 2])
                odd_len = max(1, odd_len)
                odd_item = "".join(
                    random.choices("abcdefghijklmnopqrstuvwxyz", k=odd_len)
                )

                items = pattern_items + [odd_item]
                random.shuffle(items)

            # Add to sequence
            text_parts.extend(["<ITEMS>"] + items + ["<OUTPUT>", odd_item])

        text_parts.append("<EOS>")
        full_text = " ".join(text_parts)

        # Tokenize and pad
        tokens = self.tokenize(full_text)
        if len(tokens) < target_length:
            tokens.extend([self.pad_token] * (target_length - len(tokens)))
        else:
            tokens = tokens[:target_length]

        return {
            "task": "odd_one_out",
            "difficulty": difficulty,
            "text": full_text,
            "tokens": tokens,
            "num_sets": num_sets,
            "set_size": set_size,
            "length": len(tokens),
        }

    def generate_fuzzy_regex_task(
        self, difficulty: str = "medium", target_length: int = 8192
    ) -> Dict[str, Any]:
        """Generate Fuzzy Regex Parsing task"""

        if difficulty == "basic":
            num_patterns = random.randint(3, 10)
            text_length = random.randint(100, 500)
        elif difficulty == "medium":
            num_patterns = random.randint(10, 50)
            text_length = random.randint(500, 2000)
        elif difficulty == "hard":
            num_patterns = random.randint(50, 200)
            text_length = random.randint(2000, 8000)
        else:  # extreme
            num_patterns = random.randint(200, 1000)
            text_length = random.randint(8000, 20000)

        text_parts = ["<FUZZY_REGEX>"]

        for pattern_idx in range(num_patterns):
            # Simple patterns for autoregressive learning
            patterns = [
                r"[0-9]+",
                r"[a-z]+",
                r"[A-Z]+",
                r"abc",
                r"xyz",
                r"[0-9]{3}",
                r"[a-z]{4}",
            ]
            pattern = random.choice(patterns)

            # Generate test text
            test_text = self._generate_test_text_for_pattern(
                pattern, text_length // num_patterns
            )

            # Find matches
            matches = self._fuzzy_match(pattern, test_text)

            # Add to sequence
            text_parts.extend(
                [
                    "<PATTERN>",
                    pattern,
                    "<TEXT>",
                    test_text,
                    "<OUTPUT>",
                    " ".join(matches) if matches else "no_matches",
                ]
            )

        text_parts.append("<EOS>")
        full_text = " ".join(text_parts)

        # Tokenize and pad
        tokens = self.tokenize(full_text)
        if len(tokens) < target_length:
            tokens.extend([self.pad_token] * (target_length - len(tokens)))
        else:
            tokens = tokens[:target_length]

        return {
            "task": "fuzzy_regex",
            "difficulty": difficulty,
            "text": full_text,
            "tokens": tokens,
            "num_patterns": num_patterns,
            "text_length": text_length,
            "length": len(tokens),
        }

    def _generate_test_text_for_pattern(self, pattern: str, length: int) -> str:
        """Generate test text for pattern matching"""
        words = []
        current_length = 0

        while current_length < length:
            # Generate different types of content
            content_type = random.choice(["number", "word", "mixed"])

            if content_type == "number":
                word = str(random.randint(1, 9999))
            elif content_type == "word":
                length_choice = random.randint(3, 12)
                word = "".join(
                    random.choices(
                        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
                        k=length_choice,
                    )
                )
            else:  # mixed
                chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                word = "".join(random.choices(chars, k=random.randint(3, 15)))

            words.append(word)
            current_length += len(word) + 1

        return " ".join(words)

    def _fuzzy_match(self, pattern: str, text: str) -> List[str]:
        """Simple pattern matching"""
        try:
            matches = re.findall(pattern, text)
            return matches[:10]  # Limit matches
        except re.error:
            return []

    def generate_mixed_task_sequence(
        self, target_length: int = 131072, task_distribution: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """Generate mixed sequence of tasks for long sequence training"""

        if task_distribution is None:
            task_distribution = {
                "set_data": 0.4,
                "odd_one_out": 0.3,
                "fuzzy_regex": 0.3,
            }

        all_tokens = [self.bos_token]
        task_info = []
        current_length = 1

        while current_length < target_length - 1000:
            # Choose task
            task_choice = np.random.choice(
                list(task_distribution.keys()), p=list(task_distribution.values())
            )

            difficulty = random.choice(["basic", "medium", "hard", "extreme"])
            remaining_length = target_length - current_length - 100
            task_length = min(remaining_length, random.randint(1000, 8192))

            if task_length < 500:
                break

            # Generate task
            if task_choice == "set_data":
                task_data = self.generate_set_data_task(difficulty, task_length)
            elif task_choice == "odd_one_out":
                task_data = self.generate_odd_one_out_task(difficulty, task_length)
            else:  # fuzzy_regex
                task_data = self.generate_fuzzy_regex_task(difficulty, task_length)

            # Add task tokens
            task_tokens = task_data["tokens"]
            if task_tokens[0] == self.bos_token:
                task_tokens = task_tokens[1:]
            if task_tokens[-1] == self.eos_token:
                task_tokens = task_tokens[:-1]

            all_tokens.extend(task_tokens)
            current_length += len(task_tokens)

            task_info.append(
                {
                    "task": task_choice,
                    "difficulty": difficulty,
                    "length": len(task_tokens),
                    "start_pos": current_length - len(task_tokens),
                    "end_pos": current_length,
                }
            )

            # Add separator
            all_tokens.append(self.sep_token)
            current_length += 1

        # Add final EOS
        all_tokens.append(self.eos_token)

        # Pad to exact length
        while len(all_tokens) < target_length:
            all_tokens.append(self.pad_token)

        all_tokens = all_tokens[:target_length]

        return {
            "tokens": all_tokens,
            "length": len(all_tokens),
            "task_info": task_info,
            "num_tasks": len(task_info),
            "task_distribution": task_distribution,
        }


def create_training_dataset(
    num_sequences: int = 1000, seq_length: int = 131072
) -> List[Dict]:
    """Create training dataset"""
    generator = SyntheticDataGenerator()
    dataset = []

    print(f"Generating {num_sequences} training sequences of length {seq_length}...")

    for i in range(num_sequences):
        if i % 100 == 0:
            print(f"Generated {i}/{num_sequences} sequences")

        sequence = generator.generate_mixed_task_sequence(seq_length)
        dataset.append(sequence)

    print(f"✅ Dataset generation complete: {len(dataset)} sequences")
    return dataset


if __name__ == "__main__":
    # Test data generation
    generator = SyntheticDataGenerator()

    print("🔧 Testing Synthetic Data Generation")
    print("=" * 50)

    # Test individual tasks
    set_task = generator.generate_set_data_task("medium", 2048)
    print(
        f"✅ Set Data task: {set_task['length']} tokens, {set_task['num_items']} items"
    )

    odd_task = generator.generate_odd_one_out_task("medium", 2048)
    print(
        f"✅ Odd One Out task: {odd_task['length']} tokens, {odd_task['num_sets']} sets"
    )

    regex_task = generator.generate_fuzzy_regex_task("medium", 2048)
    print(
        f"✅ Fuzzy Regex task: {regex_task['length']} tokens, {regex_task['num_patterns']} patterns"
    )

    # Test long sequence
    long_sequence = generator.generate_mixed_task_sequence(16384)
    print(
        f"✅ Mixed sequence: {long_sequence['length']} tokens, {long_sequence['num_tasks']} tasks"
    )

    print("\n🎯 Data generator ready for training!")
