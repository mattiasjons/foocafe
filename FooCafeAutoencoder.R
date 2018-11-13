library(keras)
library(zoo)
modeltrain <- read.csv("hour12.csv", stringsAsFactors=FALSE)

vibration <- modeltrain[modeltrain$SensorType==3,'Value']
vibration <- vibration[23000:length(vibration)] #take a small period of the time series...
plot(vibration, type='l')

vibration <- ((vibration-min(vibration))/(max(vibration)-min(vibration)))
vibration <- zoo::rollmean(vibration, 5)
plot(vibration, type='l')

stride = 1
x_len = 300

traindata <- lapply(seq(from=(x_len+1), to=length(vibration), by=stride), function(x) {
  vibration[(x-x_len):(x-1)]
})

traindata <- do.call(rbind, traindata)

model <- keras_model_sequential() %>% 
  layer_dense(units = 300, activation = "relu", input_shape = 300) %>% 
  layer_dense(units = 128, activation = "relu") %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 300, activation = "sigmoid")

model %>% compile(
  loss = 'mse',
  optimizer = 'adam'
)

model %>% fit(traindata, traindata, epochs = 1000, batch_size = 2000, shuffle = T)
predicted <- model %>% predict(traindata)


reconstructionError <- traindata - predicted
plot(sort(sqrt(rowMeans(reconstructionError^2))))

plot(vibration, type='l')
lines(c(predicted[1,], predicted[,300]), type='l', col="red")

plot(vibration[2000:3000], type='l')
lines(c(predicted[1,], predicted[,300])[2000:3000], type='l', col="red")

predvector <- vibration - c(predicted[1,], predicted[,300])
hist(predvector)
mean(reconstructionError)
abline(v=c(sd(predvector)*3, sd(predvector)*-3), col="red")

plot(vibration, type='l')
abline(v=which(abs(predvector) > 0.3), col="red")

