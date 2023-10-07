import numpy as np
import nltk as nl
from nltk.stem import WordNetLemmatizer
nl.download('wordnet')

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
    n = len(val_tokens)
    log_sum = 0
    for token in val_tokens:
        log_sum += np.log2(unigram_prob[token])
    perplexity = np.power(2, -log_sum/n)
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
    n = len(val_tokens)
    log_sum = 0
    for token in val_tokens:
        if token in unigram_prob.keys():
            log_sum += np.log2(unigram_prob[token])
        else:
            log_sum += np.log2(unigram_prob["<unk>"])
    perplexity = np.power(2, -log_sum/n)
    return perplexity

def get_unked_bigram_perplexity(bigram_prob, unigram_prob, val_tokens):
    n = len(val_tokens)
    log_sum = 0
    for i in range(1, len(val_tokens)):
        current_bigram = (val_tokens[i-1], val_tokens[i])
        if current_bigram in bigram_prob:
            log_sum += np.log2(bigram_prob[current_bigram])
        else:
            if (val_tokens[i-1], "<unk>") in bigram_prob:
                log_sum += np.log2(bigram_prob[(val_tokens[i-1], "<unk>")])
            elif ("<unk>", val_tokens[i]) in bigram_prob:
                log_sum += np.log2(bigram_prob[("<unk>", val_tokens[i])])
            else:
                log_sum += np.log2(bigram_prob[("<unk>", "<unk>")])
    perplexity = np.power(2, -log_sum/n)
    return perplexity


def get_addK_laplace_unigram_smoothing(k, total_word_count, unigram_count, vocab_size):
    unigram_prob = {}
    for token, count in unigram_count.items() :
        unigram_prob[token] = (count + k) / (total_word_count + (k * vocab_size))
    return unigram_prob

def get_addK_laplace_bigram_smoothing(k, bigram_count, unigram_count, vocab_size):
    bigram_prob = {}
    for token, count in bigram_count.items() :
        first_word = token[0]
        word_count = unigram_count[first_word]
        bigram_prob[token] = (count + k) / (word_count + (k * vocab_size))
    return bigram_prob



def main():
    with open("./train.txt", "r") as f:
        input_text = f.read()
    f.close()

    with open("./val.txt", "r") as f:
        val_text = f.read()
    f.close()

    # input_text = "the students like the assignment"

    lemmatizer = WordNetLemmatizer()
    lemmatize = input("Do you want to lemmatize the input text? (y/n): ")
    
    input_text = input_text.replace("\n", " ")
    input_tokens_lmt = []
    for word in input_text.split(" "):
        input_tokens_lmt.append(lemmatizer.lemmatize(word))
    input_tokens = input_text.split(" ")
    total_word_count = len(input_tokens)

    if lemmatize == "y" or lemmatize == "Y":
        input_tokens = input_tokens_lmt
        total_word_count = len(input_tokens_lmt)

    val_text = val_text.replace("\n", " ")
    val_tokens = val_text.split(" ")

    unigram_count = get_unigram_counts(input_tokens)

    # <---------------- Handling Unknown Words based on minimum frequency ----------------------->
    min_frequency = int(input("Enter the minimum frequency of a word to be considered as known word: "))
    unk_words = get_list_of_unk(min_frequency, unigram_count)
    unk_input_tokens = get_unked_tokens(unk_words, input_tokens)
    unk_vocab_size = len(set(unk_input_tokens))

    addK = int(input("Enter the value of K for Add K Smoothing: "))

    # <---------------- Unigram Language Model ----------------------->
    unk_unigram_count = get_unigram_counts(unk_input_tokens)
    unk_unigram_prob = get_unigram_probability(n_gram_count= unk_unigram_count, total_word_count= total_word_count)
    unigram_perplexity_train = get_unked_unigram_perplexity(unigram_prob= unk_unigram_prob, val_tokens= unk_input_tokens)
    unigram_perplexity = get_unked_unigram_perplexity(unigram_prob= unk_unigram_prob, val_tokens= val_tokens)
    print("Training Unigram Perplexity (Without smoothening): ", unigram_perplexity_train)
    print("Testing Unigram Perplexity (Without smoothening): ", unigram_perplexity)
    print("\n")

    addOne_laplace_smoothed_unigram_prob = get_addK_laplace_unigram_smoothing(k=1, total_word_count= total_word_count, unigram_count= unk_unigram_count, vocab_size= unk_vocab_size)
    addOne_laplace_smoothed_unigram_perplexity_train = get_unked_unigram_perplexity(unigram_prob= addOne_laplace_smoothed_unigram_prob, val_tokens= unk_input_tokens)
    addOne_laplace_smoothed_unigram_perplexity = get_unked_unigram_perplexity(unigram_prob= addOne_laplace_smoothed_unigram_prob, val_tokens= val_tokens)
    print("Training Unigram Perplexity (Laplace Smoothing): ", addOne_laplace_smoothed_unigram_perplexity_train)
    print("Testing Unigram Perplexity (Laplace Smoothing): ", addOne_laplace_smoothed_unigram_perplexity)
    print("\n")

    addK_laplace_smoothed_unigram_prob = get_addK_laplace_unigram_smoothing(k=addK, total_word_count= total_word_count, unigram_count= unk_unigram_count, vocab_size= unk_vocab_size)
    addK_laplace_smoothed_unigram_perplexity_train = get_unked_unigram_perplexity(unigram_prob= addK_laplace_smoothed_unigram_prob, val_tokens= unk_input_tokens)
    addK_laplace_smoothed_unigram_perplexity = get_unked_unigram_perplexity(unigram_prob= addK_laplace_smoothed_unigram_prob, val_tokens= val_tokens)
    print("Training Unigram Perplexity (Add K=2 Smoothing): ", addK_laplace_smoothed_unigram_perplexity_train)
    print("Testing Unigram Perplexity (Add K=2 Smoothing): ", addK_laplace_smoothed_unigram_perplexity)
    print("\n")

    # <---------------- Bigram Language Model ----------------------->
    unk_bigram_count = get_bigram_count(unk_input_tokens)
    unk_bigram_prob = get_addK_laplace_bigram_smoothing(k=1, bigram_count=unk_bigram_count, unigram_count= unk_unigram_count, vocab_size= unk_vocab_size)
    bigram_perplexity_train = get_unked_bigram_perplexity(bigram_prob= unk_bigram_prob, unigram_prob=unk_unigram_prob , val_tokens= unk_input_tokens)
    bigram_perplexity = get_unked_bigram_perplexity(bigram_prob= unk_bigram_prob, unigram_prob=unk_unigram_prob , val_tokens= val_tokens)
    print("Training Bigram Perplexity (Laplace Smoothing):", bigram_perplexity_train)
    print("Testing Bigram Perplexity (Laplace Smoothing):", bigram_perplexity)
    print("\n")

    unk_bigram_prob = get_addK_laplace_bigram_smoothing(k=addK, bigram_count=unk_bigram_count, unigram_count= unk_unigram_count, vocab_size= unk_vocab_size)
    bigram_perplexity_train = get_unked_bigram_perplexity(bigram_prob= unk_bigram_prob, unigram_prob=unk_unigram_prob , val_tokens= unk_input_tokens)
    bigram_perplexity = get_unked_bigram_perplexity(bigram_prob= unk_bigram_prob, unigram_prob=unk_unigram_prob , val_tokens= val_tokens)
    print("Traning Bigram Perplexity (Add k=2 Smoothing):", bigram_perplexity_train)
    print("Testing Bigram Perplexity (Add k=2 Smoothing):", bigram_perplexity)
    print("\n")

if  __name__ == "__main__":
    main()

