import nltk, math, numpy, collections, pdb, sys, pickle
import matplotlib.pyplot as plt

class Model():
    def __init__(self, tmat, emat, vocab, num_classes):
        self.tmat = tmat
        self.emat = emat
        self.vocab = vocab
        self.num_classes = num_classes

def q1(freq_1gram, freq_2gram):
    q1_wc = {'colorless':0, 'green':0, 'ideas':0, 'sleep':0, 'furiously':0, '$$':0}
    q1_biwc = {('colorless', 'green'):0, ('green','ideas'):0, ('ideas','sleep'):0, ('sleep','furiously'):0, ('furiously','$$'):0}
    uni_mle = 0
    for key in q1_wc:
        q1_wc[key] = freq_1gram[key]
        uni_mle += math.log(freq_1gram[key]/freq_1gram.N())
    print("Word count for- colorless green ideas sleep furiously", q1_wc)
    for key in q1_biwc:
        q1_biwc[key] = freq_2gram[key]
    return round(uni_mle,3)

def get_testcases():
    test_cases = []
    with open(TEST_CASES_FILE) as f:
        for line in f:
            arr = line.split("||")
            if len(arr) == 1:
                test_cases.append(arr[0].strip().lower())
            elif len(arr) == 2:
                test_cases.append((arr[0].strip().lower(), arr[1].strip().lower()))
    return test_cases

def run_tests(tests, tmat, emat, vocab, num_classes):
    input("\nHit 'enter' to continue...(test sample sentences using trained model)")
    c = 0
    for test in tests:
        if isinstance(test, tuple):
            sent, sent_rev = test[0], test[1]
        else:
            sent, sent_rev = test, " ".join(reversed(test.split()))
        
        _ll = getsentence_ll(sent, tmat, emat, vocab, num_classes)
        _ll_rev = getsentence_ll(sent_rev, tmat, emat, vocab, num_classes)
        print('test #' + repr(c) + ':log p(' + sent + '):'+ repr(round(_ll, 3)) + '  -vs-  log p(' + sent_rev + '):' + repr(round(_ll_rev,3)))
        print('Probability ratio:',round(math.exp(_ll - _ll_rev), 3), '\n')
        c += 1

def plot_stats(toklls, win_pr):
    plt.figure()
    for tokll, col in toklls:
        plt.plot(list(range(1,1+EM_ITERATIONS)), -1*tokll, col)
    plt.ylabel('-tokLL')
    plt.xlabel('EM iterations')
    
    plt.figure()
    axes = plt.gca()
    axes.set_xlim([0,1])
    plt.xlabel('winning probability assigments')
    plt.ylabel('words/count')
    plt.bar(list(win_pr.keys()), list(win_pr.values()), width=0.01)
    plt.show()
 
#for latent class 'c', find words transitioning to 'c' with highest probability, basically record c* = argmax_c P(c|w)
#now rank words in each class by their probability p(c*|w) and filter so they are amongst the 300 most common words, now choose top10
def get_stats(freq_1gram, tmat, vocab, num_classes):
    top_wrds = dict(freq_1gram.most_common(302)).keys()
    inv_vocab = list(vocab.keys())
    
    #for each wrd find most probable class
    wrd_max_c = numpy.argmax(tmat, 0)   #max class index
    wrd_max_pr = numpy.amax(tmat, 0)    #its probability
    #pdb.set_trace()
    win_pr = {}
    for c in range(num_classes):
        ind = numpy.where(wrd_max_c == c)
        if ind[0].size < 1:
            print('No words assigned to class: ',c)
            continue
        wrds_in_c = numpy.array([inv_vocab[x] for x in ind[0].tolist()]) #get all words transition to 'c'
        wrds_pr = wrd_max_pr[ind[0]]   #their probability
        
        srt_ind = numpy.argsort(wrds_pr)
        srt_ind = srt_ind[::-1]

        srt_wrds_c = wrds_in_c[srt_ind].tolist()
        srt_wrd_pr = wrds_pr[srt_ind].tolist()
        
        topk = {}
        for x in range(len(srt_wrds_c)):
            w, pr = srt_wrds_c[x], srt_wrd_pr[x]
            if w in top_wrds and w != '$$' and w != '@@':
                topk[w] = pr

                round_pr = round(pr, 2)
                if round_pr not in win_pr:
                    win_pr[round_pr] = 0
                win_pr[round_pr] += 1
                
            #limit to top 20 words per class
            if len(topk) >= 20:
                break
        print('class',c,':',list(topk.keys()),'\n')
    return win_pr

def initialize(num_vocab, K):
    #random initialization
    emat = numpy.random.rand(num_vocab, K) #V x C matrix
    tmat = numpy.random.rand(K, num_vocab) #C x V matrix
    #emat = emat*0.9 + 0.1
    #tmat = tmat*0.9 + 0.1
    #normalize
    er, ec = numpy.shape(emat)
    tr, tc = numpy.shape(tmat)

    for col in range(tc):
        su = sum(tmat[:,col])
        tmat[:,col] /= su

    for col in range(ec):
        su = sum(emat[:,col])
        emat[:,col] /= su
    
    checkemat = numpy.around(sum(emat), 10) == numpy.ones((K, ))
    assert numpy.all(checkemat), 'parameters of emission matrix(V x C) no well-formed'
    
    checktmat = numpy.around(sum(tmat), 10) == numpy.ones(( num_vocab, ))
    assert numpy.all(checktmat), 'parameters of transition matrix(C x V) not well-formed'

    #p(c|EOS) = na, p(BOS|c) = 0
    tmat[:, 1] = numpy.nan
    emat[0, :] = 0
    return tmat, emat

def loglikelihood(freq_2gram, tmat, emat, vocab, num_classes):
    ll = 0
    for bg, count in freq_2gram.items():
        w1, w2 = bg
        pr = 0
        for c in range(num_classes):            
            pr += emat[vocab[w2], c] * tmat[c, vocab[w1]]
        if pr > 0:
            ll += math.log(pr)*count
    return ll

def getsentence_ll(sent, tmat, emat, vocab, num_classes):
    toks =['@@']
    toks.extend(sent.split(" "))
    toks.append('$$')
    freq_2gram = nltk.FreqDist(list(nltk.bigrams(toks)))
    return loglikelihood(freq_2gram, tmat, emat, vocab, num_classes)

def estep(freq_2gram, tmat, emat, posterior, K, vocab):
    #accross all 'valid bigrams' compute p(c|w_1,w_2)
    ctr = 0
    for bg, count in freq_2gram.items():
        w1, w2 = bg
        posterior_vals = numpy.zeros(K)
        w1_i, w2_i = vocab[w1], vocab[w2]
        normalizer = 0
        #for each possible class 
        for c in range(K):
            posterior_vals[c] = emat[w2_i, c]*tmat[c, w1_i]
            normalizer += posterior_vals[c]
        #normalize by dividing with sum across all classes
        posterior[bg] = posterior_vals/normalizer
        assert round(sum(posterior[bg]), 7) == 1, 'posterior not well formed'
        
        ctr += 1
        if ctr % 10000 == 0:
            print('.',end='')
            sys.stdout.flush()

    return posterior
    
def mstep(freq_1gram, cfreq_2gram, cfreq_2gram_w2, tmat, emat, posterior, K, vocab):
    num_vocab = len(vocab)
    #compute P(c|w_1) update tmat (transition matrix)
    for w1, w1_i in vocab.items():
        #p(c|END_TOK) = NaN *no transition from end token to any class*
        if w1 == '$$':
            tmat[:,1] = numpy.nan
            continue
        if w1_i % 10000 == 0:
            print('.',end='')
            sys.stdout.flush()
        normalizer = 0
        for c in range(K):
            expected_bg_count = 0
            for w2, bg_count in cfreq_2gram[w1].items():
                bg = (w1,w2) #w_1=END eliminated at top itself
                expected_bg_count += bg_count * posterior[bg][c]
            tmat[c, w1_i] = expected_bg_count
            normalizer += expected_bg_count
        tmat[:,w1_i] /= normalizer
        assert math.fabs((freq_1gram[w1] - normalizer)) < 1, 'expected count way too off from word count'
        
    sum_tmat = numpy.around(sum(tmat), 7)
    sum_tmat_nonan = numpy.append(sum_tmat[:1], sum_tmat[2:])
    checktmat = sum_tmat_nonan == numpy.ones(( num_vocab - 1, ))
    assert numpy.all(checktmat), 'parameters of transition matrix(C x V) not well-formed'
    
    #compute P(w_2|c) update emat (emission matrix)
    w2_expc = {}
    for c in range(K):
        normalizer = 0
        for w2, w2_i in vocab.items():
            #there are no bigrams with w_2 as second word
            if w2 not in cfreq_2gram_w2:
                emat[w2_i, c] = 0
                continue
            
            #no class can generate the START_TOK, assume it always to be there
            if w2 == '@@':
                emat[0, c] = 0
                continue
            
            if w2_i % 10000 == 0:
                print('.',end='')
                sys.stdout.flush()

            expected_bg_count = 0
            for w1 in cfreq_2gram_w2[w2]:
                bg = (w1, w2)
                bg_count = cfreq_2gram_w2[w2][w1]
                posterior_pr = posterior.get(bg, numpy.zeros(K))
                expected_bg_count += bg_count * posterior_pr[c]

            emat[w2_i, c] = expected_bg_count
            normalizer += expected_bg_count

            w2_expc[w2_i] = w2_expc.get(w2_i, 0) + expected_bg_count
        emat[:, c] /= normalizer
        
    sum_emat = numpy.around(sum(emat), 7)
    checkemat = sum_emat == numpy.ones((K, ))
    assert numpy.all(checkemat), 'parameters of emission matrix(V x C) no well-formed'

    for w2 in cfreq_2gram_w2:
        w2_sum = 0
        for w in cfreq_2gram_w2[w2]:
            w2_sum += cfreq_2gram_w2[w2][w]
        assert math.fabs((w2_expc[vocab[w2]] - w2_sum)) < 1, 'expected count way too off from word count'

    return tmat, emat

def run_em(freq_2gram, cfreq_2gram, cfreq_2gram_w2, freq_1gram, vocab, num_classes, emitr):
    #EM algorithm implementation
    posterior = {}
    toklls = numpy.zeros((emitr, ))
    tmat, emat = initialize(len(vocab), num_classes)

    for it in range(emitr):
        print('Iteration:', it + 1,end='')
        posterior = estep(freq_2gram, tmat, emat, posterior, num_classes, vocab)
        tmat, emat = mstep(freq_1gram, cfreq_2gram, cfreq_2gram_w2, tmat, emat, posterior, num_classes, vocab)
        logll = loglikelihood(freq_2gram, tmat, emat, vocab, num_classes)
        toklls[it] = logll/freq_1gram.N()
        print(' marginal tokLL:',round(toklls[it],3))
        print('')

    return tmat, emat, toklls

def conditional_count_bigrams(list_bgs):
    cond_count_w2 = {}
    for bg in list_bgs:
        w1, w2 = bg
        if w2 not in cond_count_w2:
            cond_count_w2[w2] = {}
        if w1 not in cond_count_w2[w2]:
            cond_count_w2[w2][w1] = 0
        cond_count_w2[w2][w1] += 1
    return cond_count_w2

def check_params(freq_1gram, freq_2gram, cfreq_2gram, cfreq_2gram_w2):
    for bg, count in freq_2gram.items():
        w1,w2 = bg
        if w1 == '$$':
            print('invalid bigram', bg)
        elif w2 == '@@':
            print('invalid bigram', bg)
        elif count < 1:
            print('invalid bigram', bg)

    assert cfreq_2gram['$$'].N() == 0, 'there shld be no bigrams w1=END_TOK'
    assert len(cfreq_2gram_w2.get('@@', {})) == 0, 'there shld be no bigrams w2=START_TOK'

def test():
    with open(MDL_PARAMS_FILE, 'rb') as input:
        mdl = pickle.load(input)
        tests = get_testcases()
        run_tests(tests, mdl.tmat, mdl.emat, mdl.vocab, mdl.num_classes)
        print("that's all folks!")

def train(num_classes, emitr):
    num_sents, pos, END_TOK, START_TOK = 0, 0, '$$', '@@'
    bigrams_list, toks, vocab = [], [], collections.OrderedDict()
    
    vocab[START_TOK] = pos
    pos += 1
    vocab[END_TOK] = pos
    pos += 1
    print('Reading corpus')
    for fileid in nltk.corpus.brown.fileids():
        for sent in nltk.corpus.brown.sents(fileid):
            num_sents += 1
            if num_sents % 1000 == 0:
                print('.', end='')
                sys.stdout.flush()
            sent_tok = [START_TOK]
            for wrd in sent:
                n_wrd = wrd.strip().lower()
                sent_tok.append(n_wrd)
                if n_wrd not in vocab:
                    vocab[n_wrd] = pos
                    pos += 1
            sent_tok.append(END_TOK)
            
            toks.extend(sent_tok)
            bigrams_list.extend(list(nltk.bigrams(sent_tok)))
    
    freq_1gram = nltk.FreqDist(toks)    
    freq_2gram = nltk.FreqDist(bigrams_list)
    cfreq_2gram = nltk.ConditionalFreqDist(bigrams_list)
    cfreq_2gram_w2 = conditional_count_bigrams(bigrams_list)
    
    check_params(freq_1gram, freq_2gram, cfreq_2gram, cfreq_2gram_w2)

    print('\n---Corpus Info---')
    print('Number of sentences:',num_sents)
    print('Number of tokens(without END/START):',len(toks) - num_sents*2)
    print('Vocabulary size:',len(vocab))
    input("\nHit 'enter' to continue...(prints Q1)")
    print('Q1.1 log likelihood for sentence assuming unigram:', q1(freq_1gram, freq_2gram),'\n')
    input("Hit 'enter' to continue...(starts training EM)")
    
    tmat_1, emat_1, tokll_1  = run_em(freq_2gram, cfreq_2gram, cfreq_2gram_w2, freq_1gram, vocab, num_classes, emitr)

    input("Hit 'enter' to continue...(prints stats and plots)")
    win_pr = get_stats(freq_1gram,tmat_1,vocab,num_classes)
    print("close both the plots to continue")
    plot_stats([(tokll_1,'g')], win_pr)

    with open(MDL_PARAMS_FILE, 'wb') as output:
        mdl = Model(tmat_1, emat_1, vocab, num_classes)
        pickle.dump(mdl, output, pickle.HIGHEST_PROTOCOL)

LATENT_CLASSES, EM_ITERATIONS = 3, 20
TEST_CASES_FILE = 'test_sentences.txt'
MDL_PARAMS_FILE = 'model_params.pkl'

print('\n HW1 by Harshal Godhia \n')
#train(LATENT_CLASSES, EM_ITERATIONS)
test()