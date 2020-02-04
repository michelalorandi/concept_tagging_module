import math
import os
import argparse
import shutil

lexicon = {}

dict_tags = {}
dict_words = {}
dict_words_tags = {}
dict_probabilities = {}

dict_words_ner = {}
dict_tags_ner = {}
dict_words_tags_ner = {}
dict_probabilities_ner = {}

test_words = []
test_tags = []

# list that contains all the sentences separated by a blank line, each line has the following sintax:
# {word} {tag}
sentences_words_tags = []
# list of sentences (strings)
complete_sentences = []

tags_set = set()

directory_name = ''


# load the training set and count the words, word-tags, tags
def load_train(path, dict_w, dict_t, dict_w_t, sentences_file_name, improvement, iob=False):
    train = open(path, 'r')
    sentences = open(sentences_file_name, 'w')
    tmp = []
    for line in train:
        lista = line.split()
        if len(lista) != 0:
            # add the given word to the specified dictionary and count the frequency
            add_to_dict(lista[0], dict_w)

            if lista[1] == 'O' and (improvement == 1 or improvement == 2):
                # add the given tag to the specified dictionary and count the frequency
                add_to_dict(lista[0], dict_t)
                add_word_tag(lista[0], lista[0], dict_w_t)
                add_to_dict(lista[0], lexicon)
                tmp.append(lista[0])
            else:
                # add the given tag to the specified dictionary and count the frequency
                add_to_dict(lista[1], dict_t)
                add_word_tag(lista[0], lista[1], dict_w_t)
                add_to_dict(lista[0], lexicon)
                add_to_dict(lista[1], lexicon)
                tmp.append(lista[1])
                if iob:
                    tags_set.add(lista[1])
        else:
            # write the tag sentence in the file
            sentences.write(' '.join(tmp) + '\n')
            tmp = []
    sentences.close()
    train.close()


# add the given word to the specified dictionary
def add_to_dict(word, dict):
    if dict.get(word) is None:
        dict.update({word: 0})
    dict[word] += 1


# add the given word-tag to the specified dictionary and count the frequency
def add_word_tag(word, tag, dict):
    word_tag = word + '\t' + tag
    if dict.get(word_tag) is None:
        dict.update({word_tag: 0})
    dict[word_tag] += 1


# print in a file the lexicon (contains each word accepted by our system)
def create_lexicon():
    file_lexicon = open("lexicon.lex", "w")
    index = 0
    file_lexicon.write('<eps>\t' + str(index) + '\n')
    index += 1
    file_lexicon.write('<unk>\t' + str(index) + '\n')
    index += 1

    for l in lexicon:
        file_lexicon.write(l + '\t' + str(index) + '\n')
        index += 1

    file_lexicon.close()


# calculate probabilities (-log(# of times word_tag appear / # of times tag appear)) of each word_tag
def calculate_probabilities(dict_w_t, dict_t, dict_prob):
    for tag_word in dict_w_t:
        word, tag = tag_word.split()
        final_prob = -math.log(float(dict_w_t[tag_word]) / dict_t[tag])
        dict_prob.update({tag_word: final_prob})


# print in a file the tagger, each line is composed by: start_state	end_state	word_tag	probability
# the tagger contains all the occurrences of the words-tags dictionary and one occurrence of unknown word (<unk>) for each tag
def create_tagger(file_name, dict_prob, dict_t, ner=False):
    file_tagger = open(file_name + '.txt', "w")
    for word_tag in dict_prob:
        file_tagger.write('0\t0\t' + word_tag + '\t' + str(dict_prob[word_tag]) + '\n')
    if ner:
        file_tagger.write('0\t0\t<unk>\t<unk>\t1\n')
    else:
        prob = float(1) / len(dict_t)
        for tag in dict_t:
            file_tagger.write('0\t0\t<unk>\t' + tag + '\t' + str(prob) + '\n')
    file_tagger.write('0')
    file_tagger.close()

    # compile the tagger
    cmd = "fstcompile --isymbols=lexicon.lex --osymbols=lexicon.lex " + file_name + ".txt > " + file_name + ".fst"
    os.system(cmd)


# create the lm using the specified method and order
def create_lm(file_name, method, order):
    cmd = "farcompilestrings --symbols=lexicon.lex --unknown_symbol='<unk>' %s.txt > %s.far" % (file_name, file_name)
    os.system(cmd)

    cmd = "ngramcount --order=%s --require_symbols=false %s.far > %s.cnt" % (order, file_name, file_name)
    os.system(cmd)

    cmd = "ngrammake --method=%s %s.cnt > %s.lm" % (method, file_name, file_name)
    os.system(cmd)


# build the test sentences for the evaluation, so each word and each tag is added to the test_words list and the
# test_tags list respectively
# if a word is not present in the words dictionary, it is replaced with the unknown word (<unk>)
# at the end of each sentence is added an end of line (' ' blank space)
def test_sentences(method, order, improvement):
    test = open("../Datasets/NL2SparQL4NLU.test.conll.txt", 'r')
    # string that contains one sentence in words
    words_sentence = []
    # string that contains one sentence in tags
    tags_sentence = []
    for l in test:
        lista = l.split()
        if len(lista) != 0:
            if dict_words.get(lista[0]) is None:
                test_words.append('<unk>')
                words_sentence.append('<unk>')
            else:
                test_words.append(lista[0])
                words_sentence.append(lista[0])

            tags_sentence.append(lista[1])
        # at the end of the sentence compile fst
        else:
            print(' '.join(words_sentence))
            if improvement == 2:
                # compile the test sentence using our lexicon
                cmd = 'echo "%s" | farcompilestrings --symbols=lexicon.lex --unknown_symbol="<unk>" --generate_keys=1 ' \
                      '--keep_symbols | farextract --filename_suffix=".fst"' % ' '.join(words_sentence)
                os.system(cmd)

                # compose the obtained fst with the ner tagger
                cmd = "fstcompose 1.fst tagger_ner.fst > 2.fst"
                os.system(cmd)

                # compose the obtained fst with the ner lm
                cmd = "fstcompose 2.fst ner_sentences.lm | fstrmepsilon | fstshortestpath | fstrmepsilon | fsttopsort | " \
                      "fstprint --isymbols=lexicon.lex --osymbols=lexicon.lex > try.txt"
                os.system(cmd)

                # extract the sentence obtained from the ner lm
                try_f = open('./try.txt', 'r')
                tmp = []
                for l in try_f:
                    li = l.split()
                    if len(li) >= 3:
                        tmp.append(li[3])
            else:
                tmp = words_sentence

            # compile the ner sentence using our lexicon
            cmd = 'echo "%s" | farcompilestrings --symbols=lexicon.lex --unknown_symbol="<unk>" --generate_keys=1 ' \
                  '--keep_symbols | farextract --filename_suffix=".fst"' % ' '.join(tmp)
            os.system(cmd)

            # compose the obtained fst with the iob tagger
            cmd = "fstcompose 1.fst tagger.fst > 3.fst"
            os.system(cmd)

            # compose the obtained fst with the iob lm
            cmd = "fstcompose 3.fst iob_sentences.lm | fstrmepsilon | fstshortestpath | fstrmepsilon | fsttopsort | " \
                  "fstprint --isymbols=lexicon.lex --osymbols=lexicon.lex > output_%s_%s.txt" % (method, order)
            os.system(cmd)

            # build the final output in order to use it for the evaluation part
            # the sintax is: {word} {real_tag} {obtained_tag}
            file_name = 'output_%s_%s.txt' % (method, order)
            file_output = open(file_name, 'r')
            index = 0
            for line in file_output:
                output_list = line.split()
                if len(output_list) >= 3:
                    if output_list[3] in tags_set:
                        final_file.write(output_list[2] + ' ' + tags_sentence[index] + ' ' + output_list[3] + '\n')
                    else:
                        final_file.write(output_list[2] + ' ' + tags_sentence[index] + ' O\n')
                    index += 1
                else:
                    final_file.write('\n')

            file_output.close()
            words_sentence = []
            tags_sentence = []
    test.close()


# run the pos tagger module
def run(method, order, improvement, tag_set_name):
    if improvement == 2:
        train_ner_path = '../Datasets/training_ner_' + tag_set_name + '.txt'
        train_iob_path = '../Datasets/training_iob_' + tag_set_name + '.txt'
        # load the ner train dataset
        load_train(train_ner_path, dict_words_ner, dict_tags_ner, dict_words_tags_ner, 'ner_sentences.txt', improvement)
    else:
        train_iob_path = "../Datasets/NL2SparQL4NLU.train.conll.txt"
    # load the iob train dataset
    load_train(train_iob_path, dict_words, dict_tags, dict_words_tags, 'iob_sentences.txt', improvement, iob=True)

    # calculate the probabilities for the iob dataset
    calculate_probabilities(dict_words_tags, dict_tags, dict_probabilities)
    if improvement == 2:
        # calculate the probabilities for the ner dataset
        calculate_probabilities(dict_words_tags_ner, dict_tags_ner, dict_probabilities_ner)

    # create the lexicon
    create_lexicon()
    if improvement == 2:
        # create the ner tagger
        create_tagger('tagger_ner', dict_probabilities_ner, dict_tags_ner, ner=True)
    # create the iob tagger
    create_tagger('tagger', dict_probabilities, dict_tags)
    # create the iob lm
    create_lm('iob_sentences', method, order)
    if improvement == 2:
        # create the ner lm
        create_lm('ner_sentences', method, order)
    # load and run the test sentences
    test_sentences(method, order, improvement)


def my_parser():
    pars = argparse.ArgumentParser(
        description='POS tagging module for movie domain.')
    # Required positional argument
    pars.add_argument('method',
                      help='The method to be used for the lm creation')
    pars.add_argument('order',
                      help='The order to be used for the lm creation')
    # Optional argument
    pars.add_argument('--improvement', default=2, type=int,
                      help='Identify the improvement to be executed. The accepted values are: 0, 1 or 2. 0 is the baseline, 1 is the first improvement and 2 is the second improvement. The default value is 2.')
    pars.add_argument('--tags_set', default='all',
                      help='The set of tags to be used for the construction of the training sets. The accepted values are: all, pers, pers_org, pers_lang, pers_ev, pers_art, pers_art_gpe, pers_art_gpe_norp, pers_norp_org_gpe. The default value is all.')

    return pars


if __name__ == "__main__":
    parser = my_parser()
    args = parser.parse_args()

    directory_name = '%s_%s' % (args.method, args.order)
    name = '../Intermediate/final_%s_%s_%s' % (args.method, args.order, str(args.improvement))

    if args.improvement == 0 or args.improvement == 1 or args.improvement == 2:
        if args.improvement == 2:
            directory_name += '_' + args.tags_set
            name += '_' + args.tags_set
        name += '.txt'

        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
        os.chdir(directory_name)

        final_file = open(name, 'w')
        run(args.method, args.order, args.improvement, args.tags_set)
        final_file.close()
        os.chdir("..")
        shutil.rmtree(directory_name)
    else:
        print('The selected improvement is not valid. The accepted values are: 0, 1 or 2. 0 is the baseline, 1 is the first improvement and 2 is the second improvement.')
