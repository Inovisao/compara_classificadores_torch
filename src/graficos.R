# Se precisar carregar pacotes adicionais, siga os exemplos abaixo 
#install.packages("psych")

library("ggplot2")
library("gridExtra")
library("plyr")
library("stringr")
library("forcats")
library("scales")
library("forcats")
library("ExpDes")
library("dplyr")
library("ExpDes.pt")
library("reshape")
library(data.table)


###########################################################
# Gera gráficos Boxplot
###########################################################

dados <- read.table('../results_dl/results.csv',sep=',',header=TRUE)
metricas <- list("precision", "recall", "fscore")
graficos <- list()

precision <- max(dados[, c("precision")])
recall <- max(dados[, c("recall")])
fscore <- max(dados[, c("fscore")])
upper_limits <- data.frame(precision, recall, fscore)

precision <- min(dados[, c("precision")])
recall <- min(dados[, c("recall")])
fscore <- min(dados[, c("fscore")])
lower_limits <- data.frame(precision, recall, fscore)

for (lr in unique(dados$learning_rate)) {
  i <- 1
  plot.new()
  data <- dados[dados$learning_rate == lr,]
  
  for (metrica in metricas) {
  
     print(metrica)
     #metrica_str <- sprintf("%s", metrica)
     TITULO = sprintf("Architectures X Optimizers, lr = %f - %s", lr, metrica)
     g <- ggplot(data, aes_string(x="architecture", y=metrica,fill="optimizer")) + 
     geom_boxplot()+ ylim(lower_limits[,metrica] - 0.01, upper_limits[,metrica] + 0.01) +
     scale_fill_brewer(palette="Purples")+
     labs(title=TITULO,x="Architectures", y = metrica,fill="Optimizers")+
     theme(plot.title = element_text(hjust = 0.5))
     
     graficos[[i]] <- g
     i = i + 1
  }
  
  g <- grid.arrange(grobs=graficos, ncol = 1)
  ggsave(paste("../results_dl/boxplot", sub("0.", "_" ,sprintf("%f", lr)) ,".png", sep=""),g, width = 10, height = 8)
  print(g)
  
}

###########################################################
# Aplica teste Anova de 3 fatores e teste de skott-knott
###########################################################


dados <- read.table('../results_dl/results.csv',sep=',',header=TRUE)

sink('../results_dl/three_way.txt')

cat(sprintf('\n\n====>>> TESTANDO: PRECISÃO =============== \n\n',metrica))
fat3.dic(dados$learning_rate, dados$architecture, dados$optimizer, dados$precision, quali = c(TRUE, TRUE, TRUE), mcomp="sk") 
cat(sprintf('\n\n====>>> TESTANDO: RECALL ================= \n\n',metrica))
fat3.dic(dados$learning_rate, dados$architecture, dados$optimizer, dados$recall, quali = c(TRUE, TRUE, TRUE), mcomp="sk") 
cat(sprintf('\n\n====>>> TESTANDO: FSCORE ================= \n\n',metrica))
fat3.dic(dados$learning_rate, dados$architecture, dados$optimizer, dados$fscore, quali = c(TRUE, TRUE, TRUE), mcomp="sk") 


sink()

###########################################################
# Gera arquivo com algumas estatísticas
###########################################################

sink('../results_dl/statistics.txt')

dt <- data.table(dados)
cat("\n[ Estatísticas para precision]-----------------------------\n")
dt[, list(median=median(precision), IQR=IQR(precision), mean=mean(precision), sd=sd(recall)), by=.(learning_rate, architecture, optimizer)]
cat("\n[ Estatísticas para recall]-----------------------------\n")
dt[, list(median=median(recall), IQR=IQR(recall), mean=mean(recall), sd=sd(recall)), by=.(learning_rate, architecture, optimizer)]
cat("\n[ Estatísticas para fscore]-----------------------------\n")
dt[, list(median=median(fscore), IQR=IQR(fscore), mean=mean(fscore), sd=sd(fscore)), by=.(learning_rate, architecture, optimizer)]


sink()

################################################################
# DESENHA MATRIZ DE CONFUSÃO DA CONFIGURAÇÃO QUE TEVE
# A MAIOR MEDIANA
################################################################
medianas <- dt[, list(precision_median=median(precision), recall_median=median(recall)), by=.(learning_rate, architecture, optimizer)]

melhor <- medianas %>% filter(precision_median == max(precision_median) & recall_median == max(recall_median))

melhor_arquitetura <- toString(melhor$architecture[1])
melhor_otimizador <- toString(melhor$optimizer[1])
melhor_learning_rate <- format(melhor$learning_rate[1], scientific=FALSE)

nome_do_arquivo <- paste(melhor_arquitetura, "_", melhor_otimizador, "_", melhor_learning_rate, "_MATRIX.csv",sep="")

   
contaDobras <- dt[dt$architecture == melhor_arquitetura &
                  dt$optimizer == melhor_otimizador &
                  as.character(dt$learning_rate, scientific=FALSE) == melhor_learning_rate]

DOBRAS=nrow(contaDobras)
folds <- sprintf("fold_%d",seq(1:DOBRAS))
classes <- list.files('../data/all')

for (fold in folds) {
   matriz <- read.table(paste('../resultsNfolds/',fold,'/matrix/', sub("fold_", "", fold), "_", nome_do_arquivo, sep=""), sep=',',header=FALSE)
   filtrada = matriz[-1,-1]
   
#   Para normalizar por coluna usa a linha abaixo
#   normalizada <- sweep(filtrada, 2, colSums(filtrada), FUN="/")
   normalizada <- filtrada/sum(filtrada)
   
   if(fold == "fold_1") { matriz_media <- normalizada } else {matriz_media <- matriz_media + normalizada}
      
   arredondado <- round(normalizada, 2)
   colnames(arredondado) <- classes
   comNomes <- cbind(classes,arredondado)

   cm<-reshape2::melt(comNomes)

   cm <- cm %>%
      mutate(variable = factor(variable), # alphabetical order by default
             classes = factor(classes, levels = rev(unique(classes)))) # force r
   
   g<-ggplot(cm, aes(x=variable,y=classes, fill=value)) + 
      geom_tile()+xlab("Measured")+ylab("Predicted")+
      ggtitle(paste("Confusion Matrix -",melhor_arquitetura,
                    "+",melhor_otimizador, "- LR =", melhor_learning_rate, "(",fold,")"))+
      labs(fill = "Scale")+
      geom_text(aes(label = value)) +
      #scale_fill_gradient(low = "white", high = "red")+
      theme(axis.text.x = element_text(angle = 60, hjust = 1), aspect.ratio=1)
   
   ggsave(paste('../results_dl/',melhor_arquitetura,"_",melhor_otimizador,"_", melhor_learning_rate, "_", fold,'_cm.png',sep=""),g, width=6, height=5, limitsize = FALSE)
   print(g)
}

matriz_media <- matriz_media / DOBRAS
arredondado <- round(matriz_media, 2)
colnames(arredondado) <- classes
comNomes <- cbind(classes,arredondado)
cm<-reshape2::melt(comNomes)
cm <- cm %>%
   mutate(variable = factor(variable), # alphabetical order by default
          classes = factor(classes, levels = rev(unique(classes)))) # force r

g <- ggplot(cm, aes(variable, classes, fill=value)) + geom_tile() + xlab("Measured") + ylab("Predicted") + ggtitle(paste("Confusion Matrix -",melhor_arquitetura, "+",melhor_otimizador, "- LR =", melhor_learning_rate, "(mean)")) + labs(fill = "Scale") + geom_text(aes(label = value)) + theme(axis.text.x = element_text(angle = 60, hjust = 1))

ggsave(paste('../results_dl/', melhor_arquitetura, "_",melhor_otimizador, "_", melhor_learning_rate, '_MEAN_cm.png',sep=""),g, scale = 1, width=6, height=5)
print(g)


