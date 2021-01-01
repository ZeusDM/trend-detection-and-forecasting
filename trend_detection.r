library("ggplot2")
library("zoo")
library("xts")
library("prodlim")
library("igraph")
library("class")
library("reshape2")

setwd("/home/zeus/Documents/ml/trend-detection-and-forecasting")

# Parameters

j <- 5
qs <- 10
ql <- 5*qs
a <- 5

# Functions

standardization <- function(v) {
  pnorm((v - mean(v))/sd(v))
}

pseudo_standardization <- function(v, vs) {
  pnorm((v - mean(vs))/sd(vs))
}

import_stock <- function(path = "./csv/ibov.csv"){
  database <- read.csv(path,
                       encoding="UTF-8",
                       header=FALSE,
                       stringsAsFactors = F)
  xts_aux <- xts(database$V2, order.by = as.Date(database$V1, "%m/%d/%Y"))
  colnames(xts_aux) <- "Raw"
  xts_aux$Smooth <- rollmeanr(xts_aux$Raw, j)
  xts_aux <- xts_aux[-(1:(j-1))]
  xts_aux
}

training_step <- function(xts_training, stability_parameter){
  # Short and Long Moving Averages
  
  xts_training$ShortAvg <- rollmeanr(xts_training$Smooth, qs)
  xts_training$LongAvg <- rollmeanr(xts_training$Smooth, ql)
  
  # Extracting features
  
  xts_training$ShortNoise <- xts_training$Smooth/xts_training$ShortAvg - 1
  xts_training$LongNoise <- xts_training$Smooth/xts_training$LongAvg - 1
  xts_training$ShortGrad <- (xts_training$ShortAvg)/c(0, xts_training$ShortAvg)[-(nrow(xts_training)+1)] - 1
  xts_training$LongGrad <- (xts_training$LongAvg)/c(0, xts_training$LongAvg)[-(nrow(xts_training)+1)] - 1
  xts_training$ShortRHL <- (xts_training$Smooth + rollmaxr(-xts_training$Smooth, qs))/(rollmaxr(xts_training$Smooth, qs) + rollmaxr(-xts_training$Smooth, qs))
  xts_training$LongRHL <- (xts_training$Smooth + rollmaxr(-xts_training$Smooth, ql))/(rollmaxr(xts_training$Smooth, ql) + rollmaxr(-xts_training$Smooth, ql))
  
  xts_training <- xts_training[-(1:(j+ql-1))]
  
  xts_training$z1 <- floor(a*standardization(xts_training$ShortNoise))
  xts_training$z2 <- floor(a*standardization(xts_training$LongNoise))
  xts_training$z3 <- floor(a*standardization(xts_training$ShortGrad))
  xts_training$z4 <- floor(a*standardization(xts_training$LongGrad))
  xts_training$z5 <- floor(a*standardization(xts_training$ShortRHL))
  xts_training$z6 <- floor(a*standardization(xts_training$LongRHL))
  
  # Network
  
  w <- data.frame(xts_training[, 11:16])
  wu <- unique(w)
  xts_training$Node <- row.match(w, wu)
  
  m <- matrix(replicate(n = nrow(wu)^2, expr = 0), nrow = nrow(wu))
  
  for(i in 2:nrow(xts_training)){
    pre <- as.numeric(xts_training$Node[i-1])
    pos <- as.numeric(xts_training$Node[i])
    if (pre != pos){
      m[pre, pos] <- m[pre, pos] + 1
    }
  }
  
  # Community Detection
  
  g <- graph_from_adjacency_matrix(m, mode = c("undirected"), weighted = TRUE)
  cluster <- cluster_fast_greedy(g)
  community <- membership(cluster)
  k <- max(community)
  
  # Trend assignment
  
  r <- replicate(k, 0)
  xts_training$Trend <- replicate(nrow(xts_training), 0)
  
  v <- c(diff(log(as.numeric(xts_training$Smooth))),0)
  
  for (i in 1:k) {
    vaux <- community[xts_training$Node] == i
    mv <- mean(v[vaux])
    if (mv >= stability_parameter) {
      r[i] <- 1
    }
    else if (mv <= -stability_parameter) {
      r[i] <- -1
    }
    else{
      r[i] <- 0
    }
  }
  
  xts_training$Trend = r[community[xts_training$Node]]
  
  # qplot(index(xts_training), coredata(xts_training$Smooth), colour = factor(coredata(xts_training$Trend)))
  
  xts_training
}

testing_step <- function(xts_testing, xts_training){
  # Trend Forecasting
  
  xts_testing$ShortAvg <- rollmeanr(xts_testing$Smooth, qs)
  xts_testing$LongAvg <- rollmeanr(xts_testing$Smooth, ql)
  
  # Extracting features
  
  xts_testing$ShortNoise <- xts_testing$Smooth/xts_testing$ShortAvg - 1
  xts_testing$LongNoise <- xts_testing$Smooth/xts_testing$LongAvg - 1
  xts_testing$ShortGrad <- (xts_testing$ShortAvg)/c(0, xts_testing$ShortAvg)[-(nrow(xts_testing)+1)] - 1
  xts_testing$LongGrad <- (xts_testing$LongAvg)/c(0, xts_testing$LongAvg)[-(nrow(xts_testing)+1)] - 1
  xts_testing$ShortRHL <- (xts_testing$Smooth + rollmaxr(-xts_testing$Smooth, qs))/(rollmaxr(xts_testing$Smooth, qs) + rollmaxr(-xts_testing$Smooth, qs))
  xts_testing$LongRHL <- (xts_testing$Smooth + rollmaxr(-xts_testing$Smooth, ql))/(rollmaxr(xts_testing$Smooth, ql) + rollmaxr(-xts_testing$Smooth, ql))
  
  xts_testing <- xts_testing[-(1:(j+ql-1))]
  
  xts_testing$z1 <- floor(a*pseudo_standardization(xts_testing$ShortNoise, xts_training$ShortNoise))
  xts_testing$z2 <- floor(a*pseudo_standardization(xts_testing$LongNoise, xts_training$LongNoise))
  xts_testing$z3 <- floor(a*pseudo_standardization(xts_testing$ShortGrad, xts_training$ShortGrad))
  xts_testing$z4 <- floor(a*pseudo_standardization(xts_testing$LongGrad, xts_training$LongGrad))
  xts_testing$z5 <- floor(a*pseudo_standardization(xts_testing$ShortRHL, xts_training$ShortRHL))
  xts_testing$z6 <- floor(a*pseudo_standardization(xts_testing$LongRHL, xts_training$LongRHL))
  
  x <- knn(xts_training[,11:16], xts_testing[,11:16], xts_training$Trend)
  xts_testing$Trend <- as.numeric(levels(x)[x])
  #qplot(index(xts_testing), coredata(xts_testing$Smooth), colour = factor(coredata(xts_testing$Trend)))
  
  xts_testing #output
}

plot_confmatrix <- function(xts_testing, stock_name, stability_parameter){
  v <- c(diff(log(as.numeric(xts_testing$Raw))), 0)
  cm <- matrix(replicate(9, 0), nrow = 3)
  for (i in 1:length(xts_testing$Raw)) {
    if (v[i] >= stability_parameter) {
      actual <- 1
    }
    else if (v[i] <= -stability_parameter) {
      actual <- -1
    }
    else{
      actual <- 0
    }
    predicted <- xts_testing$Trend[i]
    cm[predicted+2, actual+2] <- cm[predicted+2, actual+2] + 1
  }
  colnames(cm) <- c(-1, 0, 1)
  rownames(cm) <- c(-1, 0, 1)
  
  melted_cm <- melt(cm)
  colnames(melted_cm) <- c("Predicted", "Actual", "value")
  
  ggheatmap <- ggplot(melted_cm, aes(Actual, Predicted, fill = value))+
    geom_tile(color = "white")+
    scale_fill_gradient2(high = "red", low = "white", 
                         midpoint = 0, space = "Lab", 
                         name="") +
    theme_minimal()+ # minimal theme
    ggtitle(paste("Backtest of", stock_name,"on raw data\nStability:",exp(-stability_parameter),"~",exp(stability_parameter))) +
    coord_fixed()
  
  ggheatmap + 
    geom_text(aes(Actual, Predicted, label = value), color = "black", size = 4) +
    theme(
      axis.title.x = element_blank(),
      axis.title.y = element_blank(),
      panel.grid.major = element_blank(),
      panel.border = element_blank(),
      panel.background = element_blank(),
      axis.ticks = element_blank(),)+
    guides(fill = guide_colorbar(barwidth = 1, barheight = 7,
                                 title.position = "top", title.hjust = 0.5))
  
}

backtest <- function(xts_testing){
  # Backtest
  cash <- replicate(nrow(xts_testing), 0)
  inv <- replicate(nrow(xts_testing), 0)
  #cash[1] <- xts_testing$Smooth[1]
  cash[1] <- xts_testing$Raw[1]
  inv[1] <- 0
  cur_state <- 0
  for (t in 2:nrow(xts_testing)) {
    #op_price <- (as.numeric(xts_testing$Smooth[t]) + as.numeric(xts_testing$Smooth[t-1]))/2
    op_price <- (as.numeric(xts_testing$Raw[t]) + as.numeric(xts_testing$Raw[t-1]))/2
    if (xts_testing$Trend[t] == 1){
      if (cur_state == 0){
        cur_state <- 1
        cash[t] <- cash[t-1] - op_price
        inv[t] <- inv[t-1] + op_price
      }
      else if (cur_state == 1){
        cash[t] <- cash[t-1]
        inv[t] <- op_price
      }
      else {
        cur_state <- 0
        cash[t] <- cash[t-1] - op_price
        inv[t] <- 0
      }
    }
    else if (xts_testing$Trend[t] == -1){
      if (cur_state == 0){
        cur_state <- -1
        cash[t] <- cash[t-1] + op_price
        inv[t] <- inv[t-1] - op_price
      }
      else if (cur_state == 1){
        cur_state <- 0
        cash[t] <- cash[t-1] + op_price
        inv[t] <- 0
      }
      else {
        cash[t] <- cash[t-1]
        inv[t] <- -op_price
      }
    }
    else{
      if (cur_state == 0){ # do nothing
        cash[t] <- cash[t-1]
      }
      else if (cur_state == 1){
        cur_state <- 0
        cash[t] <- cash[t-1] + op_price
        inv[t] <- 0
      }
      else {
        cur_state <- 0
        cash[t] <- cash[t-1] - op_price
        inv[t] <- 0
      }
    }
  }
  xts_testing$Balance <- cash + inv 
  
  xts_testing
}

plot_backtest <- function(xts_testing, stock_name, best_validation_parameter){
  #qplot(index(xts_testing), coredata(xts_testing$Smooth), colour = factor(coredata(xts_testing$Trend)))
  #qplot(index(xts_testing), coredata(xts_testing$Balance), colour = factor(coredata(xts_testing$Trend)))
  
  bt_plot <- ggplot() +
    geom_point(mapping = aes(x = index(xts_testing), y = coredata(xts_testing$Raw), colour = factor(coredata(xts_testing$Trend)))) +
    labs(colour = "Trend Assignment") +
    geom_point(mapping = aes(x = index(xts_testing), y = coredata(xts_testing$Balance)), colour="gray20") +
    ggtitle(paste("Backtest of", stock_name,"on raw data\nStability:",exp(-best_validation_parameter),"~",exp(best_validation_parameter))) +
    xlab("Time") + ylab("Value")
  
  print(bt_plot)
}

main <- function(stock_name){
  xts1 <- import_stock(paste("./csv/", stock_name, ".csv", sep = ""))
  
  training_threshold <- floor(nrow(xts1)*0.8)
  validation_threshold <- floor(training_threshold*0.7)
  
  xts_training <- xts1[1:validation_threshold]
  xts_validation <- xts1[(validation_threshold+1):training_threshold]
  xts_testing <- xts1[(training_threshold+1):nrow(xts1)]
  #xts_testing2 <- xts1[1:training_threshold]
  
  best_validation_parameter <- 0
  best_validation_value <- 0
  
  ## TODO: trocar por binary search
  for (stability_parameter in (exp(0:20/12000) - 1)){
    xts_training1 <- training_step(xts_training, stability_parameter)
    xts_validation1 <- testing_step(xts_validation, xts_training1)
    xts_validation1 <- backtest(xts_validation1)
    if(xts_validation1$Balance[nrow(xts_validation1)] > best_validation_value){
      best_validation_parameter <- stability_parameter
      best_validation_value <- xts_validation1$Balance[nrow(xts_validation1)]
    }
  }
  
  print(best_validation_parameter)
  xts_training2 <- training_step(xts_training, best_validation_parameter)
  xts_testing2 <- testing_step(xts_testing, xts_training2)
  #xts_testing2 <- testing_step(xts_testing2, xts_training)
  
  xts_testing2 <- backtest(xts_testing2)
  plot_backtest(xts_testing2, stock_name, best_validation_parameter)
  
  plot_confmatrix(xts_testing2, stock_name, best_validation_parameter)
  
  #xts_testing2 <- backtest(xts_testing2)
  #plot_backtest(xts_testing2)
}

main("spx")
main("ibov")
main("indu")
main("ndx")

