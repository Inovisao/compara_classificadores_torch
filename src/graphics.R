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

# Define nicknames for the architectures. Keep the indexes symmetric!
original_arch_name <- list("alexnet",
                           "coat_tiny",
                           "convnext_base", 
                           "lambda_resnet26rpt_256", 
                           "lamhalobotnet50ts_256",
                           "maxvit_rmlp_tiny_rw_256",
                           "resnet18",
                           "sebotnet33ts_256",
                           "swinv2_base_window16_256",
                           "vgg19",
                           "vit_relpos_base_patch32_plus_rpn_256",
                           "default_siamese")

arch_nickname <- list("alexnet",
                      "coat",
                      "convnext",
                      "LambdaResnet",
                      "LamHaloBotNet",
                      "MaxViT",
                      "ResNet18",
                      "SEBotNet",
                      "SwinV2",
                      "Vgg19",
                      "ViTRelPosRPN",
                      "default_siamese")


original_optim_name <- list("sgd", "adam", "adagrad")
optim_nickname <- list("SGD", "Adam", "Adagrad")

###########################################################
# Get the data and prepare the dataframe.
###########################################################

# Read the csv containing the data.
data <- read.table('../results_dl/results.csv', sep=',', header=TRUE)
data$original_arch_names <- data$architecture
data$original_optim_names <- data$optimizer

# Change the columns with aesthetically acceptable names.
for (i in seq(length(original_arch_name))) {
  data[data$architecture == original_arch_name[i], c("architecture")] <- arch_nickname[i]
}

for (i in seq(length(original_optim_name))) {
  data[data$optimizer == original_optim_name[i], c("optimizer")] <- optim_nickname[i]
}

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

one_way_anova <- function(dataframe, factors) {
  # Applies one-way ANOVA and Tukey's HSD test to the factor in the list for each metric.
  # The response variables are precision, recall, and fscore.
  sink("../results_dl/one_way.txt")
  
  factor_name <- sprintf("%s", factors[1])
  
  # One-way ANOVA and Tukey's HSD test for precision
  cat(sprintf('\n\n====>>> TESTING: PRECISION for %s =============== \n\n', factor_name))
  aov_result_precision <- aov(precision ~ ., data = subset(dataframe, select = c(factor_name, "precision")))
  print(summary(aov_result_precision))
  cat("\n\n====>>> Tukey's HSD TEST for PRECISION =============== \n\n")
  tukey_precision <- TukeyHSD(aov_result_precision)
  print(tukey_precision)
  
  # One-way ANOVA and Tukey's HSD test for recall
  cat(sprintf('\n\n====>>> TESTING: RECALL for %s =============== \n\n', factor_name))
  aov_result_recall <- aov(recall ~ ., data = subset(dataframe, select = c(factor_name, "recall")))
  print(summary(aov_result_recall))
  cat("\n\n====>>> Tukey's HSD TEST for RECALL =============== \n\n")
  tukey_recall <- TukeyHSD(aov_result_recall)
  print(tukey_recall)
  
  # One-way ANOVA and Tukey's HSD test for fscore
  cat(sprintf('\n\n====>>> TESTING: FSCORE for %s =============== \n\n', factor_name))
  aov_result_fscore <- aov(fscore ~ ., data = subset(dataframe, select = c(factor_name, "fscore")))
  print(summary(aov_result_fscore))
  cat("\n\n====>>> Tukey's HSD TEST for FSCORE =============== \n\n")
  tukey_fscore <- TukeyHSD(aov_result_fscore)
  print(tukey_fscore)
  
  sink()
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
if (length(factors) == 1){
  one_way_anova(data, factors)
} else if (length(factors) == 2) {
  two_way_anova(data, factors)
} else if (length(factors) == 3) {
  three_way_anova(data, factors)
} else {
  print("Incorrect number of factors. Anova could not be applied.")
}

###########################################################
# Get some statistics.
###########################################################


options(width=10000) # Change line width

dt <- data.table(data)

precision_statistics <- dt[, list(median=median(precision), IQR=IQR(precision), mean=mean(precision), sd=sd(precision)), by=.(learning_rate, architecture, optimizer)]

recall_statistics <- dt[, list(median=median(recall), IQR=IQR(recall), mean=mean(recall), sd=sd(recall)), by=.(learning_rate, architecture, optimizer)]

fscore_statistics <- dt[, list(median=median(fscore), IQR=IQR(fscore), mean=mean(fscore), sd=sd(fscore)), by=.(learning_rate, architecture, optimizer)]

# Create a .txt with the statistics.
sink('../results_dl/statistics.txt')

cat("\n[ Statistics for precision ]-----------------------------\n")
print(precision_statistics)

cat("\n[ Statistics for recall]-----------------------------\n")
print(recall_statistics)

cat("\n[ Statistics for fscore]-----------------------------\n")
print(fscore_statistics)
sink()

# Save the statistics in LaTeX table format.
sink("../results_dl/statistics_for_latex.txt")

cat(kbl(precision_statistics, caption="Statistics for precision",
      format="latex",
      col.names=c("Learning rate", "Architecture", "Optimizer", "Median", "IQR", "Mean", "SD"),
      align="r"))

cat(kbl(recall_statistics, caption="Statistics for recall",
      format="latex",
      col.names=c("Learning rate", "Architecture", "Optimizer", "Median", "IQR", "Mean", "SD"),
      align="r"))

cat(kbl(fscore_statistics, caption="Statistics for fscore",
      format="latex",
      col.names=c("Learning rate", "Architecture", "Optimizer", "Median", "IQR", "Mean", "SD"),
      align="r"))

sink()
###########################################################
# Plot the confusion matrix of the configuration that achieved
# the highest median.
###########################################################

median_values <- dt[, list(precision=median(precision), recall=median(recall), fscore=median(fscore)), by=.(learning_rate, architecture, optimizer, original_arch_names, original_optim_names)]

for (metric in metrics) {
  print(sprintf("Generating best matrices for %s.", metric))
  dir.create(paste("../results_dl/matrices_for_best_", metric, sep=""))
  # Get the combination with highest precision median.
  best <- median_values %>% filter(median_values[[metric]] == max(median_values[[metric]]))

  print(best)
  # Create the filename.
  filename = sprintf("%s_%s_%s_MATRIX.csv", best$original_arch_names, best$original_optim_names, format(best$learning_rate, scientific=FALSE))

  # Get the lines of the combination across different folds.
  across_folds <- dt[dt$architecture == best$architecture &
                      dt$optimizer == best$optimizer &
                      dt$learning_rate == best$learning_rate]

  # Get the number of folds.
  num_folds <- nrow(across_folds)

  # Get a list of strings like "fold_n", a list of classes.
  folds <- sprintf("fold_%d", seq(1:num_folds))
  # Use system command to list files in the directory
  class_files <- system("ls -1 ../data/all", intern = TRUE)
  # Convert the result into a character vector
  classes <- as.vector(class_files)
  # Iterate over the folds to get the data for the matrix.
  for (fold in folds) {

    # Read the matrix for one fold.
    matrix <- read.table(paste('../resultsNfolds/', fold, '/matrix/', sub("fold_", "", fold), "_", filename, sep=""), sep=',',header=FALSE)

    # Get only the values, without class numbers.
    filtered <- matrix[-1, -1]

    # Normalize the matrix.
    normalized <- filtered / sum(filtered)

    # Accumulate the values to get a matrix with mean values.
    if (fold == "fold_1") {
      mean_matrix <- normalized
    } else {
      mean_matrix <- mean_matrix + normalized
    }

    rounded <- round(normalized, 2)
    colnames(rounded) <- classes
    with_names <- cbind(classes, rounded)

    confusion_matrix <- reshape2::melt(with_names)

    confusion_matrix <- confusion_matrix %>% mutate(variable=factor(variable),
                                                    classes=factor(classes, levels=rev(unique(classes))))


    matrix_title <- sprintf("Fold %s: %s, %s. LR = %s.", sub("fold_", "", fold), best$architecture, best$optimizer, format(best$learning_rate, scientific=TRUE))

    g <- ggplot(confusion_matrix,
                aes(x=variable,y=classes, fill=value)) +
                geom_tile() +
                xlab("Predicted") +
                ylab("Measured") +
                ggtitle(matrix_title) +
                labs(fill="Scale") +
                geom_text(aes(label=value)) +
                theme(axis.text.x=element_text(angle=60, hjust = 1), aspect.ratio=1)

    ggsave(paste('../results_dl/matrices_for_best_', metric, "/", best$architecture, "_", best$optimizer, "_", best$learning_rate, "_", fold, '_cm.png',sep=""),
           g,
           width=6,
           height=5,
           limitsize = FALSE)

    print(g)

  }

  mean_matrix <- mean_matrix / num_folds
  rounded <- round(mean_matrix, 2)
  colnames(rounded) <- classes
  with_names <- cbind(classes, rounded)
  confusion_matrix <- reshape2::melt(with_names)
  confusion_matrix <- confusion_matrix %>% mutate(variable=factor(variable), # alphabetical order by default
                                                  classes=factor(classes, levels=rev(unique(classes)))) # force r

  matrix_title <- sprintf("Matrix (mean): %s, %s. LR = %s.", best$architecture, best$optimizer, best$learning_rate)
  g <- ggplot(confusion_matrix,
              aes(variable, classes, fill=value)) +
              geom_tile() +
              xlab("Predicted") +
              ylab("Measured") +
              ggtitle(matrix_title) +
              labs(fill="Scale") +
              geom_text(aes(label=value)) +
              theme(axis.text.x=element_text(angle=60, hjust=1))

  ggsave(paste('../results_dl/matrices_for_best_', metric, "/", best$architecture, "_", best$optimizer, "_", best$learning_rate, '_MEAN_cm.png', sep=""), g, scale=1, width=6, height=5)
  print(g)
}

