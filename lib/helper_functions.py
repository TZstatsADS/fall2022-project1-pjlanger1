#generating some metadata for subsetting
def aggregate_metrics(df):
    a = df.groupby(['title','author']).size().reset_index().rename(columns={0:'count'})
    b = pd.DataFrame(a['author'].value_counts()).reset_index()
    b = b.rename(columns={"author":"count_total_works","index":"author"})
    metadata = pd.merge(a,b,on = 'author', how = 'left')
    return metadata

def create_corpus(df):
    #Loading Essential Packages to the Function
    import nltk
    from nltk.stem import PorterStemmer
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    from nltk.sentiment import SentimentIntensityAnalyzer
    nltk.download('vader_lexicon')
    
    ps = PorterStemmer() #initializing stemmer
    stop_words = stopwords.words('english') #initializing stopwords array
    
    df['author_strp'] = df['author'].apply(lambda x: re.sub('[\W_]+','_', x.lower()))
    list_author = list(df['author_strp'].unique())
    
    df['author_title_unique'] = df['author'] + ' ' + df['title']
    df['sent_count'] = df.groupby(['author_title_unique']).cumcount() + 1
    metrics = aggregate_metrics(df).drop(['author','count_total_works'],axis = 1)
    df = pd.merge(df,metrics,how = 'left', on = 'title')
    df['perc_cum_total'] = df['sent_count'] / df['count']
    df['perc_total'] = 1 / df['count']
    
    #Sentiment Analysis
    sia = SentimentIntensityAnalyzer()

    df['composite_sentiment'] = df['sentence_lowered'].apply(lambda x: sia.polarity_scores(x)['compound'])
    
    
    
    
    author_corpus = {key: [] for key in list_author}
    
    for i in range(0,len(df['tokenized_txt'])):
        string_1 = str(df['tokenized_txt'].iloc[i])[1:-1].split(',')
        author = df['author_strp'].iloc[i]
        for j in string_1:
            w = re.sub(r'\W+','',j)
            if w not in stop_words:
                new_w = ps.stem(w)
                author_corpus[author].append(new_w)
                
    return df, author_corpus


def generate_graphs(df_processed):
    agg_table_title = df_processed.pivot_table(index = ['author','title'],values = 'composite_sentiment', aggfunc = np.mean).reset_index()
    agg_table_author = df_processed.pivot_table(index = 'author',values = 'composite_sentiment', aggfunc = np.mean).reset_index()
    agg_tab_author['sentiment_rate'] = agg_tab_author['composite_sentiment'].apply(lambda x: 'negative' if x < -0.05 else 'positive' if x> 0.05 else 'neutral')

    
    #arry_plato = df_processed['composite_sentiment'][df_processed['author']=='Merleau-Ponty']
    #arry_plato2 = df_processed['perc_cum_total'][df_processed['author']=='Merleau-Ponty']
    
    #arry_plato3 = df_processed['composite_sentiment'][df_processed['author']=='Keynes']
    #arry_plato4 = df_processed['perc_cum_total'][df_processed['author']=='Keynes']
    
    #Pairwise sentiment scores
    #fig, ax = plt.subplots()
    #ax.plot(agg_table_title['title'],agg_table_title['composite_sentiment'])
    #ax.plot(arry_plato2, arry_plato, label='Prices 2008-2018', color='blue')
    #ax.plot(arry_plato4, arry_plato3, label='Prices 2010-2018', color = 'red')
    #legend = ax.legend(loc='center right', fontsize='x-small')
    #plt.xlabel('years')
    #plt.ylabel('prices')
    #plt.title('Comparison of the different prices')
    
    plt.bar(agg_table_author['author'],agg_table_author['composite_sentiment'])
    plt.xticks(rotation=90)
    plt.title('Average Composite Sentiment Score (SIA) by Philosopher')
    plt.show()
    
    return agg_table_title, agg_table_author

def dictionary_parser(x):
    #import gensim
    from gensim import corpora, models
    from collections import Counter

    listicle = []
    for key in x:
        listicle.append(x[key])

    dictionary_LDA = corpora.Dictionary(listicle)
    corpus = [dictionary_LDA.doc2bow(list_of_tokens) for list_of_tokens in listicle]
    num_topics = 7
    lda_model = models.LdaModel(corpus, num_topics=num_topics, \
                                  id2word=dictionary_LDA, \
                                  passes=4, alpha=[0.01]*num_topics, \
                                  eta=[0.01]*len(dictionary_LDA.keys()))

    for i,topic in lda_model.show_topics(formatted=True, num_topics=num_topics, num_words=10):
        print(str(i)+": "+ topic)
        print()
    
    #new_structure = []
    #for key in x:
    #    new_structure.append(" ".join(str(y) for y in x[key]))
    
    #count_vect = CountVectorizer(stop_words=stopwords.words('english'),lowercase=False)
    #x_counts = count_vect.fit_transform(new_structure)
    #tfidf_transformer = TfidfTransformer()
    #x_tfidf = tfidf_transformer.fit_transform(x_counts)
    
    #dimension = 10
    #lda = LDA(n_components = dimension)
    #lda_array = lda.fit_transform(x_tfidf)
    #components = [lda.components_[i] for i in range(len(lda.components_))]
    
    #features = count_vect.get_feature_names()
    
    #important_words = [sorted(features, key = lambda x: components[j][features.index(x)], reverse = True)[:5] for j in range(len(components))]


    #x_ct = x_counts.todense()
    
    get_document_topics = get_document_topics = [lda_model.get_document_topics(item) for item in corpus]
    
    return get_document_topics


def time_series_analysis(df):
    from pathlib import Path
    import os
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, acf
    from scipy.optimize import brute
    dict_acf = {}
    
    lst_works = df['title'].unique()
    #os.chdir('/figs')
    for l in lst_works:
        title = str("ACF of " + l)
        filename4 = str(os.path.join(Path.home(), "Downloads/figs/"+l+".png"))
        fig, ax = plt.subplots(1, 2, figsize=(10,5))
        data = df['composite_sentiment'][df['title']==l]
        plot_acf(data, lags = 10, title = title, ax = ax[0])
        plot_pacf(data, lags = 10, title = 'P'+ title, ax = ax[1])
        dict_acf[l] = acf(data, nlags = 10)
        
        plt.savefig(filename4)
        plt.close(fig)

        
    print("ACF/PACF - Saved to " + str(os.path.join(Path.home(), "Downloads/figs/")))
    
    
    return dict_acf
