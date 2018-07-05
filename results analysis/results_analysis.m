tbl = readtable('C:\python\deep learning course\project\Seq2Seq\matlab.xlsx');
mat = table2array(tbl);
%%

inds = (find(mat(:,1) < 50));

figure()
plot( mat(inds,1) , mat(inds,2), 'ob' )
ylim([-0.1 , 1.1]);
xlabel('# of Words in True Translation','fontsize',14,'fontweight','bold');
ylabel('BLEU Score','fontsize',14,'fontweight','bold');
title('BLEU Score as Function of # of Words in True Translation','fontsize',18,'fontweight','bold')
%%
figure()
hist( mat(inds,3),50)
xlim([0 30]);
xlabel('# of Words in English Sentence','fontsize',14,'fontweight','bold');
ylabel('Frequency','fontsize',14,'fontweight','bold');
title('# of Words in English Sentence Distribution','fontsize',18,'fontweight','bold')


