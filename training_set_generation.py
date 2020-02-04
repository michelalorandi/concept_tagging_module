import os
import spacy
import argparse

nlp = spacy.load('en_core_web_sm')

complete_sentences = []
iob_tags_sentences = []
ner_sentences = []

all_tags = ['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']
pers = ['PERSON']
pers_org = ['PERSON', 'ORG']
pers_art = ['PERSON', 'WORK_OF_ART']
pers_lang = ['PERSON', 'LANGUAGE']
pers_ev = ['PERSON', 'EVENT']
pers_art_gpe = ['PERSON', 'GPE', 'WORK_OF_ART']
pers_art_gpe_norp = ['PERSON', 'NORP', 'GPE', 'WORK_OF_ART']


def init():
    inputs = open("./Datasets/NL2SparQL4NLU.train.conll.txt", 'r')
    tmp_sent = ''
    tmp_iob = []
    for line in inputs:
        lista = line.split()
        if len(lista) != 0:
            tmp_sent += ' ' + lista[0]
            tmp_iob.append(lista[1])
        else:
            iob_tags_sentences.append(tmp_iob)
            complete_sentences.append(tmp_sent.strip())
            tmp_sent = ''
            tmp_iob = []
    inputs.close()


def truecase(sentence):
    print(sentence)
    file = open('sent_text.txt', 'w')
    file.write(sentence)
    file.close()

    # path = os.path.join(os.path.expanduser('~'), 'Downloads', 'stanford-corenlp-full-2016-10-31', '*')
    path = './stanford-corenlp-full-2016-10-31/*'
    cmd = "java -cp \"" + path + "\" edu.stanford.nlp.pipeline.StanfordCoreNLP -outputFormat conll -annotators tokenize,ssplit,truecase -file sent_text.txt -truecase.overwriteText"
    os.system(cmd)

    file_truecase = open("sent_text.txt.conll", "r")
    tmp = []
    for line in file_truecase:
        s = line.split('\t')
        if len(s) > 1:
            tmp.append(s[1])

    sent = ' '.join(tmp)
    doc = nlp(unicode(sent))
    sentence2 = ''
    start = 0
    for ent in doc.ents:
        tmp = ent.text.split()
        label = ''
        for l in range(len(tmp)):
            label += ' ' + ent.label_
        sentence2 += sentence[start: ent.start_char:] + label.strip()
        start = ent.end_char
    sentence2 += sent[start::]
    return sentence2


def save_training_sets(tags, set_name):
    output_ner = open('./Datasets/training_ner_' + set_name + '.txt', 'w')
    output_iob = open('./Datasets/training_iob_' + set_name + '.txt', 'w')
    tot = len(complete_sentences)
    sent_index = 0
    for sentence in complete_sentences:
        sent = truecase(sentence)
        sent = sent.split()
        sentence = sentence.split()
        i = 0
        print(str(sent_index) + '/' + str(tot))
        print(sentence)
        print(iob_tags_sentences[sent_index])
        for w in sentence:
            if sent[i] in tags:
                output_ner.write(w + '\t' + sent[i] + '\n')
                output_iob.write(sent[i] + '\t' + iob_tags_sentences[sent_index][i] + '\n')
            else:
                output_ner.write(w + '\t' + w + '\n')
                output_iob.write(w + '\t' + iob_tags_sentences[sent_index][i] + '\n')
            i += 1
        output_ner.write('\n')
        output_iob.write('\n')
        sent_index += 1
    output_ner.close()
    if os.path.isfile('sent_text.txt'):
        os.remove('sent_text.txt')
    if os.path.isfile('sent_text.txt.conll'):
        os.remove('sent_text.txt.conll')


def my_parser():
    pars = argparse.ArgumentParser(
        description='Module for the generation of the training set for the POS tagging module for movie domain.')
    pars.add_argument('tags_set',
                      help='The set of tags to be used for the construction of the training sets. The accepted values are: all, pers, pers_org, pers_lang, pers_ev, pers_art, pers_art_gpe, pers_art_gpe_norp.')
    return pars


if __name__ == "__main__":
    parser = my_parser()
    args = parser.parse_args()

    init()

    if args.tags_set == 'all':
        save_training_sets(all_tags, 'all')
    elif args.tags_set == 'pers':
        save_training_sets(pers, 'pers')
    elif args.tags_set == 'pers_lang':
        save_training_sets(pers_lang, 'pers_lang')
    elif args.tags_set == 'pers_ev':
        save_training_sets(pers_ev, 'pers_ev')
    elif args.tags_set == 'pers_art':
        save_training_sets(pers_art, 'pers_art')
    elif args.tags_set == 'pers_art_gpe':
        save_training_sets(pers_art_gpe, 'pers_art_gpe')
    elif args.tags_set == 'pers_art_gpe_norp':
        save_training_sets(pers_art_gpe_norp, 'pers_art_gpe_norp')
    else:
        print('The specified value is not valid. The accepted values are: all, pers, pers_org, pers_lang, pers_ev, pers_art, pers_art_gpe, pers_art_gpe_norp')
