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
library("kableExtra")
library(data.table)
# If needed, use the command below, changing the name of the package.
#install.packages("kableExtra")

# Read the csv containing the data.
data <- read.table('../results_dl/results.csv', sep=',', header=TRUE)

# Define shorter names for the architectures. Add more lines for new architectures.
data[data$architecture == "lambda_resnet26rpt_256", c("architecture")] <- "lambda_resnet"
data[data$architecture == "lamhalobotnet50ts_256", c("architecture")] <- "lamhalobotnet"
data[data$architecture == "maxvit_rmlp_tiny_rw_256", c("architecture")] <- "maxvit"
data[data$architecture == "sebotnet33ts_256", c("architecture")] <- "sebotnet"
data[data$architecture == "swinv2_base_window16_256", c("architecture")] <- "swinv2"
data[data$architecture == "vit_relpos_base_patch32_plus_rpn_256", c("architecture")] <- "vit_relpos_rpn"


###########################################################
# Create boxplots
###########################################################

metrics <- list("precision", "recall", "fscore")
plots <- list()

# Find the highest and the lowest observed values for each metric.
precision <- max(data[, c("precision")])
recall <- max(data[, c("recall")])
fscore <- max(data[, c("fscore")])
upper_limits <- data.frame(precision, recall, fscore)

precision <- min(data[, c("precision")])
recall <- min(data[, c("recall")])
fscore <- min(data[, c("fscore")])
lower_limits <- data.frame(precision, recall, fscore)

for (lr in unique(data$learning_rate)) {
  i <- 1
  plot.new()
  data_one_lr <- data[data$learning_rate == lr,]
  print(sprintf("Generating boxplots for lr = %s.", format(lr, scientific=TRUE)))
  for (metric in metrics) {
  
      print(sprintf("Metric: %s.", metric))
    
      # Create a string for the title.
      TITLE = sprintf("Architectures X Optimizers, lr = %s: %s", format(lr, scientific=TRUE), metric)
     
      # Create the boxplot.
      g <- ggplot(data_one_lr, aes_string(x="architecture", y=metric,fill="optimizer")) + 
        geom_boxplot() + 
        ylim(lower_limits[,metric] - 0.01, upper_limits[,metric] + 0.01) +
        scale_fill_brewer(palette="Purples") +
        labs(title=TITLE, x="Architectures", y=metric, fill="Optimizers") +
        theme(plot.title=element_text(hjust = 0.5))
     
      # Append the boxplot to a list, to create the full image later.
      plots[[i]] <- g
      i = i + 1
  }
  
  g <- grid.arrange(grobs=plots, ncol = 1)
  ggsave(paste("../results_dl/boxplot", sub("0.", "_" ,sprintf("%f", lr)) ,".png", sep=""),g, width = 10, height = 8)
  print(g)
  
}


###########################################################
# Apply anova and skott-knott test.
###########################################################

# Verify which variables (out of architecture, optimizer and learning rate)
# have at least two values.
possible_factors <- list("architecture", "optimizer", "learning_rate")
factors <- list()
i <- 1
for (possible_factor in possible_factors) {
  if (length(unique(data[, possible_factor])) > 1) {
    factors[i] <- possible_factor
    i <- i + 1
  }
}


two_way_anova <- function(dataframe, factors) {
  # Applies two way anova to any two factors given in a list.
  # The response variables are precision, recall and fscore.
  sink("../results_dl/two_way.txt")    
  
  cat(sprintf('\n\n====>>> TESTING: PRECISION =============== \n\n'))
  fat2.dic(dataframe[, sprintf("%s", factors[1])], 
           dataframe[, sprintf("%s", factors[2])], 
           dataframe$precision, 
           quali=c(TRUE, TRUE),
           mcomp="sk")
  
  cat(sprintf('\n\n====>>> TESTING: RECALL =============== \n\n'))
  fat2.dic(dataframe[, sprintf("%s", factors[1])], 
           dataframe[, sprintf("%s", factors[2])], 
           dataframe$recall, 
           quali=c(TRUE, TRUE),
           mcomp="sk") 
  
  cat(sprintf('\n\n====>>> TESTING: FSCORE =============== \n\n'))
  fat2.dic(dataframe[, sprintf("%s", factors[1])], 
           dataframe[, sprintf("%s", factors[2])], 
           dataframe$fscore, 
           quali=c(TRUE, TRUE),
           mcomp="sk")
  
  sink()
}

three_way_anova <- function(dataframe, factors) {
  # Applies three way anova to any three factors given in a list.
  # The response variables are precision, recall and fscore. 
  
  sink("../results_dl/three_way.txt")
  
  cat(sprintf('\n\n====>>> TESTING: PRECISION =============== \n\n'))
  fat3.dic(dataframe[, sprintf("%s", factors[1])], 
           dataframe[, sprintf("%s", factors[2])], 
           dataframe[, sprintf("%s", factors[3])], 
           dataframe$precision, 
           quali=c(TRUE, TRUE, TRUE), 
           mcomp="sk") 
  
  cat(sprintf('\n\n====>>> TESTING: RECALL ================= \n\n'))
  fat3.dic(dataframe[, sprintf("%s", factors[1])], 
           dataframe[, sprintf("%s", factors[2])], 
           dataframe[, sprintf("%s", factors[3])], 
           dataframe$recall, 
           quali=c(TRUE, TRUE, TRUE), 
           mcomp="sk") 
  
  cat(sprintf('\n\n====>>> TESTING: FSCORE ================= \n\n'))
  fat3.dic(dataframe[, sprintf("%s", factors[1])], 
           dataframe[, sprintf("%s", factors[2])], 
           dataframe[, sprintf("%s", factors[3])], 
           dataframe$fscore, 
           quali=c(TRUE, TRUE, TRUE), 
           mcomp="sk") 
  
  sink()
}

# Apply anova according to the number of factors.
if (length(factors) == 2) {
  two_way_anova(data, factors)
} else if (length(factors) == 3) {
  three_way_anova(data, factors)
} else {
  print("Incorrect number of factors. Anova could not be applied.")
}

###########################################################
# Get some statistics.
###########################################################

sink('../results_dl/statistics.txt')

dt <- data.table(data)

cat("\n[ Statistics for precision ]-----------------------------\n")
test_statistics <- dt[, list(median=median(precision), IQR=IQR(precision), mean=mean(precision), sd=sd(recall)), by=.(learning_rate, architecture, optimizer)]

print(test_statistics)



cat("\n[ Statistics for recall]-----------------------------\n")
dt[, list(median=median(recall), IQR=IQR(recall), mean=mean(recall), sd=sd(recall)), by=.(learning_rate, architecture, optimizer)]

cat("\n[ Statistics for fscore]-----------------------------\n")
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


