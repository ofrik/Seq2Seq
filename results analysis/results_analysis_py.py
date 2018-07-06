import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



folder_path = 'C:\\python\\deep learning course\\project v2\\Seq2Seq\\'

##### Loss plot
train_loss = pd.read_csv(folder_path + 'run_.-tag-train_loss.csv')
val_loss = pd.read_csv(folder_path + 'run_.-tag-validation_loss.csv')
test_loss = pd.read_csv(folder_path + 'run_.-tag-test_loss.csv')


fig = plt.gcf()
fig.set_size_inches(12, 8, forward=True)
train, = plt.plot(train_loss['Step'] , train_loss['Value'] ,'r'  )
plt.xlabel('Step Number')
plt.ylabel('Loss')
plt.title('Train Set Loss as Function of Step Number')
plt.rcParams.update({'font.size': 20})
plt.show()

fig = plt.gcf()
fig.set_size_inches(12, 8, forward=True)
val, = plt.plot(val_loss['Step'] , val_loss['Value'] , 'g' )
plt.xlabel('Step Number')
plt.ylabel('Loss')
plt.title('Validation Set Loss as Function of Step Number')
plt.rcParams.update({'font.size': 20})
plt.show()

fig = plt.gcf()
fig.set_size_inches(12, 8, forward=True)
test, = plt.plot(test_loss['Step'] , test_loss['Value'] ,'b'  )
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.title('Test Set Loss as Function of Epoch Number')
plt.rcParams.update({'font.size': 20})
plt.show()

##### BLEU plot
val_bleu = pd.read_csv(folder_path + 'run_.-tag-validation_bleu.csv')
test_bleu = pd.read_csv(folder_path + 'run_.-tag-test_bleu.csv')


fig = plt.gcf()
fig.set_size_inches(12, 8, forward=True)
val, = plt.plot(val_bleu['Step'] , val_bleu['Value'] , 'g' )
plt.xlabel('Step Number')
plt.ylabel('BLEU Score')
plt.title('Validation Set BLEU Score as Function of Step Number')
plt.rcParams.update({'font.size': 20})
plt.show()

fig = plt.gcf()
fig.set_size_inches(12, 8, forward=True)
test, = plt.plot(test_bleu['Step'] , test_bleu['Value'] ,'b'  )
plt.xlabel('Epoch Number')
plt.ylabel('BLEU Score')
plt.title('Test Set BLEU Score as Function of Epoch Number')
plt.rcParams.update({'font.size': 20})
plt.show()



##### sentences length plot
sent_length = pd.read_csv(folder_path + 'test_results.csv')
count = sent_length['english_sentence'].str.count(' ').add(1)
sent_length['eng_length'] = count

count2 = sent_length['expected_translation'].str.count(' ').add(1)
sent_length['true_length'] = count2


fig = plt.gcf()
fig.set_size_inches(12, 8, forward=True)
val, = plt.plot(sent_length['eng_length'] , sent_length['bleu_score'] , 'b.' )
plt.xlabel('# of Words in English Sentence')
plt.ylabel('BLEU Score')
plt.title('BLEU Score as Function of # of Words in English Sentence')
plt.rcParams.update({'font.size': 20})
plt.show()

fig = plt.gcf()
fig.set_size_inches(12, 8, forward=True)
val, = plt.plot(sent_length['true_length'] , sent_length['bleu_score'] , 'b.' )
plt.xlabel('# of Words in True Translation')
plt.ylabel('BLEU Score')
plt.title('BLEU Score as Function of # of Words in True Translation')
plt.rcParams.update({'font.size': 20})
plt.show()


fig = plt.gcf()
fig.set_size_inches(12, 8, forward=True)
plt.hist(sent_length['eng_length'] , 50, normed=1, facecolor='blue', alpha=0.75)
plt.xlabel('# of Words in English Sentence')
plt.ylabel('Frequency')
plt.title('# of Words in English Sentence Distribution')
plt.rcParams.update({'font.size': 20})
plt.show()

fig = plt.gcf()
fig.set_size_inches(12, 8, forward=True)
plt.hist(sent_length['true_length'] , 50, normed=1, facecolor='blue', alpha=0.75)
plt.xlabel('# of Words in True Translation')
plt.ylabel('Frequency')
plt.title('# of Words in True Translation Distribution')
plt.rcParams.update({'font.size': 20})
plt.show()


##### embedding plot

sent_emb = pd.read_csv(folder_path + 'test_encoding.csv')
test_sent_emb = pd.read_csv(folder_path + 'test_encoding2.csv')
sent_emb = pd.concat([sent_emb,test_sent_emb])


pca = PCA(n_components=2)
sent_emb['encoding'] = sent_emb['encoding'].str[1:-1]
sent_emb['encoding_num'] = sent_emb['encoding'].apply(lambda x: np.fromstring( x, dtype=float,count=64, sep=' '))

tmp = np.matrix(sent_emb['encoding_num'].tolist())
principalComponents = pca.fit_transform(tmp)
exp_var = pca.explained_variance_ratio_.cumsum()[0:1]
exp_var =  round(float(np.float64(100*exp_var)), 4)

#sim1a = 17918
#sim1b = 17919
#
#sim2a = 17950
#sim2b = 17951
#
#dif1a = 17712
#
#fig = plt.gcf()
#ax = fig.add_subplot(111)
#fig.set_size_inches(12, 8, forward=True)
#for 
#plt.plot(principalComponents[sim1a,0] , principalComponents[sim1a,1] , 'ro')
#ax.annotate(sent_emb['english_sentence'].iloc[sim1a] , (principalComponents[sim1a,0] , principalComponents[sim1a,1]) )
#plt.plot(principalComponents[sim1b,0] , principalComponents[sim1b,1] , 'ro')
#ax.annotate(sent_emb['english_sentence'].iloc[sim1b] , (principalComponents[sim1b,0] , principalComponents[sim1b,1]) )
#
#plt.plot(principalComponents[sim2a,0] , principalComponents[sim2a,1] , 'bo')
#ax.annotate(sent_emb['english_sentence'].iloc[sim2a] , (principalComponents[sim2a,0]-1 , principalComponents[sim2a,1]) )
#plt.plot(principalComponents[sim2b,0] , principalComponents[sim2b,1] , 'bo')
#ax.annotate(sent_emb['english_sentence'].iloc[sim2b] , (principalComponents[sim2b,0]-0.2 , principalComponents[sim2b,1]) )
#
#plt.plot(principalComponents[dif1a,0] , principalComponents[dif1a,1] , 'bo')
#ax.annotate(sent_emb['english_sentence'].iloc[dif1a] , (principalComponents[dif1a,0] , principalComponents[dif1a,1]) )
#
#
#plt.xlabel('Principal Component 1')
#plt.ylabel('Principal Component 2')
#plt.title('Embedding Analysis Using 2 PCs - ' + str( exp_var) + '%' )
#plt.rcParams.update({'font.size': 20})
#plt.show()

#inds1 = range(len(principalComponents)-12 , len(principalComponents)-6)
inds1 = range(len(principalComponents)-6 , len(principalComponents))

fig = plt.gcf()
ax = fig.add_subplot(111)
fig.set_size_inches(12, 8, forward=True)
for ind in inds1:
    plt.plot(principalComponents[ind,0] , principalComponents[ind,1] , 'ro')
    ax.annotate(sent_emb['english_sentence'].iloc[ind] , (principalComponents[ind,0] , principalComponents[ind,1]) )

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
#ax.set_xlim(-18, 5)
ax.set_xlim(1, 25)
plt.title('Embedding Analysis Using 2 PCs - ' + str( exp_var) + '%' )
plt.rcParams.update({'font.size': 20})
plt.show()





