import numpy as np
import os

def get_unigram_counts(input_tokens):
    n_gram_count = {}
    for token in input_tokens:
        if token in n_gram_count:
            n_gram_count[token] += 1
        else:
            n_gram_count[token] = 1
    return n_gram_count

def get_unigram_probability(n_gram_count, total_word_count):
    unigram_prob = {}
    for token, count in n_gram_count.items():
        unigram_prob[token] = count / total_word_count
    return unigram_prob

def get_bigram_count(input_tokens):
    bigram_count = {}
    for i in range(1, len(input_tokens)):
        current_bigram = (input_tokens[i-1], input_tokens[i])
        # current_text = ' '.join(input_tokens[i-1:i+1])
        if current_bigram in bigram_count:
            bigram_count[current_bigram] += 1
        else:
            bigram_count[current_bigram] = 1
    return bigram_count

def get_bigram_probability(bigram_count, unigram_count):
    bigram_prob = {}
    for bigram, count in bigram_count.items():
        first_word = bigram[0]
        word_count = unigram_count[first_word]
        bigram_prob[bigram] = count / word_count
    return bigram_prob

def get_unigram_perplexity(unigram_prob, val_tokens):
    N = len(val_tokens)
    log_sum = 0
    for token in val_tokens:
        log_sum += np.log2(unigram_prob[token])
    perplexity = np.power(2, -log_sum/N)
    return perplexity

def get_list_of_unk(min_frequency, n_gram_counts):
    unk_words = []
    for token, count in n_gram_counts.items():
        if count <= min_frequency:
            unk_words.append(token)
    return unk_words


def get_unked_tokens(unk_words, input_tokens):
    unked_tokens = []
    for token in input_tokens:
        if token in unk_words:
            unked_tokens.append("<unk>")
        else:
            unked_tokens.append(token)
    return unked_tokens

def get_unked_unigram_perplexity(unigram_prob, val_tokens):
    N = len(val_tokens)
    log_sum = 0
    for token in val_tokens:
        if token in unigram_prob:
            log_sum += np.log2(unigram_prob[token])
        else:
            log_sum += np.log2(unigram_prob["<unk>"])
    perplexity = np.power(2, -log_sum/N)
    return perplexity


def main():
    with open("./train.txt", "r") as f:
        input_text = f.read()
    f.close()

    with open("./val.txt", "r") as f:
        val_text = f.read()
    f.close()

    # input_text = "the students like the assignment"
    
    input_text = input_text.replace("\n", " ")
    input_tokens = input_text.split(" ")
    total_word_count = len(input_tokens)

    val_text = val_text.replace("\n", " ")
    val_tokens = val_text.split(" ")


    unigram_count = get_unigram_counts(input_tokens)
    vocab_size = len(unigram_count.keys())
    unigram_prob = get_unigram_probability(n_gram_count= unigram_count, total_word_count= total_word_count)

    bigram_count = get_bigram_count(input_tokens)
    bigram_prob = get_bigram_probability(bigram_count= bigram_count, unigram_count= unigram_count)

    # TODO: Calculate perplexity for bigram models

    # unigram_perplexity = get_unigram_perplexity(unigram_prob= unigram_prob, val_tokens= val_tokens)
    # print(unigram_perplexity)

    # print(sorted(unigram_count.values()))
    min_frequency = 5
    unk_words = get_list_of_unk(min_frequency, unigram_count)
    unk_input_tokens = get_unked_tokens(unk_words, input_tokens)

    unk_unigram_count = get_unigram_counts(unk_input_tokens)
    unk_unigram_prob = get_unigram_probability(n_gram_count= unk_unigram_count, total_word_count= total_word_count)

    unk_bigram_count = get_bigram_count(unk_input_tokens)
    unk_bigram_prob = get_bigram_probability(bigram_count= unk_bigram_count, unigram_count= unk_unigram_count)

    unigram_perplexity = get_unked_unigram_perplexity(unigram_prob= unk_unigram_prob, val_tokens= val_tokens)
    # print(unigram_perplexity)

    # TODO: Calculate perplexity for bigram models with UNK


















if  __name__ == "__main__":
    main()

